import json

import pytest

import app
from conftest import KB_DIR
from knowledge import KnowledgeBase, load_from_dir
from llm import FakeLLM


class FakeTable:
    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[Item["ticket_id"]] = Item

    def get_item(self, Key):
        item = self.items.get(Key["ticket_id"])
        return {"Item": item} if item else {}


class FakeSNS:
    def __init__(self):
        self.messages = []

    def publish(self, **kwargs):
        self.messages.append(kwargs)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALERT_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:alerts")
    table, sns = FakeTable(), FakeSNS()
    app._clients.clear()
    app._clients.update({"table": table, "sns": sns, "llm": FakeLLM(),
                         "kb": KnowledgeBase(load_from_dir(KB_DIR))})
    yield table, sns
    app._clients.clear()


def post(body):
    return app.create_ticket({"body": json.dumps(body)}, None)


def test_create_ticket_saves_and_returns_201(env):
    table, sns = env
    resp = post({"subject": "Outlook not syncing", "description": "No new email since 9am",
                 "requester": "sam@example.com"})
    assert resp["statusCode"] == 201
    body = json.loads(resp["body"])
    assert body["priority"] == "P3" and body["category"] == "email_collaboration"
    assert body["ticket_id"] in table.items
    assert sns.messages == []


def test_p1_sends_alert(env):
    _, sns = env
    resp = post({"subject": "Ransomware?", "description": "Files are being renamed and a ransom note appeared"})
    assert json.loads(resp["body"])["priority"] == "P1"
    assert len(sns.messages) == 1


def test_personal_data_not_sent_to_model(env, monkeypatch):
    seen = {}

    class SpyLLM(FakeLLM):
        def complete(self, system, user, max_tokens=800):
            seen["user"] = user
            return super().complete(system, user, max_tokens)

    app._clients["llm"] = SpyLLM()
    post({"subject": "Locked out", "description": "Email me at jane@example.com or call 301-555-0142"})
    assert "jane@example.com" not in seen["user"] and "301-555-0142" not in seen["user"]


def test_missing_fields_400(env):
    assert post({"subject": "hi"})["statusCode"] == 400


def test_bad_json_400(env):
    assert app.create_ticket({"body": "{not json"}, None)["statusCode"] == 400


def test_ai_failure_still_saves_ticket(env):
    class BrokenLLM:
        def complete(self, *a, **k):
            raise RuntimeError("Bedrock throttled")

    app._clients["llm"] = BrokenLLM()
    resp = post({"subject": "Printer", "description": "Printer offline"})
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 201 and body["needs_human_review"] is True


def test_get_ticket(env):
    ticket_id = json.loads(post({"subject": "VPN", "description": "VPN won't connect"})["body"])["ticket_id"]
    resp = app.get_ticket({"pathParameters": {"ticket_id": ticket_id}}, None)
    assert resp["statusCode"] == 200
    missing = app.get_ticket({"pathParameters": {"ticket_id": "00000000-0000-0000-0000-000000000000"}}, None)
    assert missing["statusCode"] == 404
    assert app.get_ticket({"pathParameters": {"ticket_id": "abc"}}, None)["statusCode"] == 400
