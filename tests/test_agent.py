import json
import time
import uuid

import pytest

import app
from agent import MAX_STEPS, HelpDeskTools, run_agent
from conftest import KB_DIR
from knowledge import KnowledgeBase, load_from_dir
from llm import FakeLLM
from store import MemoryTable, MemoryTicketStore

NOW = 1_800_000_000


def make_ticket(table, subject, description="", priority="P2", minutes_ago=10, **extra):
    ticket_id = str(uuid.uuid4())
    table.put_item(Item={"ticket_id": ticket_id, "created_at": NOW - minutes_ago * 60, "subject": subject,
                         "description": description, "priority": priority, "category": "network",
                         "status": "open", "review_reasons": [], **extra})
    return ticket_id


@pytest.fixture
def setup():
    table = MemoryTable()
    alerts = []
    tools = HelpDeskTools(MemoryTicketStore(table), KnowledgeBase(load_from_dir(KB_DIR)),
                          notifier=lambda tid, reason: alerts.append((tid, reason)), now=lambda: NOW)
    return table, tools, alerts


class ScriptedLLM:
    """Replays a fixed list of model turns, and records what it was sent."""

    provider = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, system, messages, tools, max_tokens=800):
        self.seen.append(json.loads(json.dumps(messages)))
        turn = self.turns.pop(0)
        return {"content": turn, "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}


def use(name, args, call_id="t1"):
    return [{"type": "tool_use", "id": call_id, "name": name, "input": args}]


def text(t):
    return [{"type": "text", "text": t}]


# ---- the loop ---------------------------------------------------------------------------

def test_agent_calls_tool_then_answers(setup):
    _, tools, _ = setup
    llm = ScriptedLLM([use("search_knowledge_base", {"query": "vpn"}), text("Follow vpn.md.")])
    result = run_agent("How do I fix the VPN?", llm, tools)
    assert result["answer"] == "Follow vpn.md." and result["stopped"] == "done"
    assert result["steps"][0]["tool"] == "search_knowledge_base" and result["steps"][0]["ok"]
    tool_result = llm.seen[1][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "vpn.md" in tool_result["content"]
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10}


def test_step_limit_stops_runaway_loop(setup):
    _, tools, _ = setup
    llm = ScriptedLLM([use("search_knowledge_base", {"query": "vpn"}, f"t{i}") for i in range(MAX_STEPS)])
    result = run_agent("loop forever", llm, tools)
    assert result["stopped"] == "step_limit" and "technician should review" in result["answer"]
    assert len(result["steps"]) == MAX_STEPS


def test_unknown_tool_and_bad_input_return_errors_not_crashes(setup):
    _, tools, _ = setup
    llm = ScriptedLLM([use("delete_all_tickets", {}), use("get_ticket", {"ticket_id": "abc"}, "t2"), text("ok")])
    result = run_agent("try things", llm, tools)
    assert [s["ok"] for s in result["steps"]] == [False, False]
    assert "Unknown tool" in result["steps"][0]["error"] and "UUID" in result["steps"][1]["error"]


def test_pii_redacted_from_question_and_tool_results(setup):
    table, tools, _ = setup
    tid = make_ticket(table, "Locked out", "Call me at 301-555-0142 or jane@example.com")
    llm = ScriptedLLM([use("get_ticket", {"ticket_id": tid}), text("done")])
    result = run_agent("Ticket from bob@example.com, what now?", llm, tools)
    everything = json.dumps(llm.seen)
    assert "bob@example.com" not in everything and "jane@example.com" not in everything
    assert "301-555-0142" not in everything and result["redacted_counts"] == {"email": 1}


# ---- tools and their guardrails ----------------------------------------------------------

def test_search_recent_tickets_finds_pattern(setup):
    table, tools, _ = setup
    for _ in range(3):
        make_ticket(table, "VPN failing")
    make_ticket(table, "VPN old", minutes_ago=60 * 48)  # outside the window
    make_ticket(table, "Printer jam")
    out, err = tools.run("search_recent_tickets", {"keyword": "VPN", "hours": 24})
    assert not err and out["count"] == 3


def test_escalate_to_p1_alerts_once(setup):
    table, tools, alerts = setup
    tid = make_ticket(table, "VPN failing")
    out, err = tools.run("escalate_ticket", {"ticket_id": tid, "reason": "possible outage", "priority": "P1"})
    assert not err and out["priority"] == "P1"
    assert table.items[tid]["priority"] == "P1" and table.items[tid]["needs_human_review"] is True
    assert alerts == [(tid, "possible outage")]
    # Second escalation in the same request is blocked (no alert spam).
    _, err = tools.run("escalate_ticket", {"ticket_id": tid, "reason": "again", "priority": "P1"})
    assert err and len(alerts) == 1


def test_priority_can_never_be_lowered(setup):
    table, tools, alerts = setup
    tid = make_ticket(table, "Ransomware note", priority="P1")
    out, err = tools.run("escalate_ticket", {"ticket_id": tid, "reason": "user says it's fine", "priority": "P4"})
    assert not err and out["priority"] == "P1" and "can't be lowered" in out["note"]
    assert table.items[tid]["priority"] == "P1" and alerts == []


def test_notes_are_capped_and_redacted(setup):
    table, tools, _ = setup
    tid = make_ticket(table, "Outlook")
    for i in range(3):
        assert not tools.run("add_ticket_note", {"ticket_id": tid, "note": f"note {i} call 301-555-0142"})[1]
    assert tools.run("add_ticket_note", {"ticket_id": tid, "note": "one too many"})[1]
    notes = table.items[tid]["agent_notes"]
    assert len(notes) == 3 and "301-555-0142" not in json.dumps(notes)


def test_tool_failure_is_reported_not_raised(setup):
    _, tools, _ = setup

    class BrokenStore:
        def recent(self, since):
            raise RuntimeError("DynamoDB unavailable")

    tools.store = BrokenStore()
    out, err = tools.run("search_recent_tickets", {"keyword": "vpn"})
    assert err and "failed" in out["error"]


def test_fake_model_detects_outage_end_to_end(setup):
    table, tools, alerts = setup
    for minutes in (5, 15, 25):
        make_ticket(table, "VPN won't connect", minutes_ago=minutes)
    result = run_agent("Is there a VPN outage right now?", FakeLLM(), tools)
    assert [s["tool"] for s in result["steps"]] == ["search_recent_tickets", "search_knowledge_base",
                                                     "escalate_ticket"]
    assert result["actions"][0]["priority_after"] == "P1" and len(alerts) == 1
    assert "escalated" in result["answer"]


# ---- the API route ---------------------------------------------------------------------

@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("ALERT_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:alerts")
    table = MemoryTable()
    sent = []

    class SNS:
        def publish(self, **kw):
            sent.append(kw)

    app._clients.clear()
    app._clients.update({"table": table, "store": MemoryTicketStore(table), "sns": SNS(),
                         "llm": FakeLLM(), "kb": KnowledgeBase(load_from_dir(KB_DIR))})
    yield table, sent
    app._clients.clear()


def ask(body):
    resp = app.agent({"body": json.dumps(body)}, None)
    return resp["statusCode"], json.loads(resp["body"])


def test_agent_route_escalates_and_emails(api):
    table, sent = api
    now = int(time.time())
    for i in range(3):
        table.put_item(Item={"ticket_id": str(uuid.uuid4()), "created_at": now - i * 60,
                             "subject": "VPN down", "description": "", "priority": "P2", "status": "open"})
    status, body = ask({"message": "Any VPN problems today?"})
    assert status == 200 and body["stopped"] == "done" and len(sent) == 1
    assert "latency_ms" in body and body["model_provider"] == "fake"


def test_agent_route_validates_input(api):
    assert ask({})[0] == 400
    assert ask({"message": "x" * 2001})[0] == 400
    assert ask({"message": "hi", "ticket_id": "not-a-uuid"})[0] == 400


def test_agent_route_when_model_is_down(api):
    class DownLLM:
        provider = "down"

        def chat(self, *a, **k):
            raise RuntimeError("throttled")

    app._clients["llm"] = DownLLM()
    status, body = ask({"message": "Any VPN problems?"})
    assert status == 503 and "unavailable" in body["error"]


# ---- DynamoDB store (what runs in AWS) ----------------------------------------------------

def test_dynamo_store_scan_pages_and_update_expression():
    from store import DynamoTicketStore

    class Table:
        def __init__(self):
            self.calls = []

        def scan(self, **kw):
            self.calls.append(kw)
            if "ExclusiveStartKey" not in kw:
                return {"Items": [{"ticket_id": "a"}], "LastEvaluatedKey": {"ticket_id": "a"}}
            return {"Items": [{"ticket_id": "b"}]}

        def update_item(self, **kw):
            self.calls.append(kw)

    table = Table()
    store = DynamoTicketStore(table)
    assert [t["ticket_id"] for t in store.recent(100)] == ["a", "b"]
    store.update("a", {"priority": "P1"}, note={"note": "x"})
    update = table.calls[-1]
    assert update["UpdateExpression"].startswith("SET #f0 = :v0")
    assert "list_append(if_not_exists(#notes, :empty), :note)" in update["UpdateExpression"]
    assert update["ExpressionAttributeNames"] == {"#f0": "priority", "#notes": "agent_notes"}
    assert update["ConditionExpression"] == "attribute_exists(ticket_id)"
