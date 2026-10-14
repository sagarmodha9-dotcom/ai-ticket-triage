import json

from conftest import KB_DIR
from knowledge import KnowledgeBase, load_from_dir
from llm import FakeLLM
from triage import apply_guardrails, parse_model_output, triage_ticket, validate

KB = KnowledgeBase(load_from_dir(KB_DIR))


class StubLLM:
    """Returns whatever text we tell it to, so we can test bad model output."""
    def __init__(self, text):
        self.text = text

    def complete(self, system, user, max_tokens=800):
        return self.text


def test_parses_json_inside_code_fence():
    assert parse_model_output('```json\n{"priority": "P2"}\n```') == {"priority": "P2"}


def test_garbage_output_goes_to_human():
    result = triage_ticket("Printer jam", "Paper stuck", StubLLM("sorry I can't"), KB)
    assert result["needs_human_review"] is True
    assert result["priority"] == "P3"


def test_invalid_values_are_corrected():
    result, problems = validate({"priority": "urgent", "category": "stuff", "confidence": "high"})
    assert result["priority"] == "P3" and result["category"] == "other"
    assert result["confidence"] == 0.0
    assert len(problems) == 3


def test_security_words_raise_priority_even_if_model_says_low():
    low = json.dumps({"priority": "P4", "category": "email_collaboration", "summary": "x",
                      "suggested_reply": "y", "confidence": 0.9, "sources": []})
    result = triage_ticket("Weird email", "I clicked a link and it asked for my password, maybe phishing",
                           StubLLM(low), KB)
    assert result["priority"] == "P2"
    assert result["category"] == "security"
    assert result["needs_human_review"] is True


def test_made_up_sources_are_removed():
    out = json.dumps({"priority": "P2", "category": "network", "summary": "VPN down",
                      "suggested_reply": "Restart the client", "confidence": 0.9,
                      "sources": ["vpn.md", "imaginary.md"]})
    result = triage_ticket("VPN down", "VPN won't connect from home", StubLLM(out), KB)
    assert result["sources"] == ["vpn.md"]


def test_low_confidence_flagged():
    result, _ = validate({"priority": "P3", "category": "hardware", "confidence": 0.3})
    assert apply_guardrails(result, "printer", [])["needs_human_review"] is True


def test_fake_llm_end_to_end():
    result = triage_ticket("Locked out", "I'm locked out after too many password attempts", FakeLLM(), KB)
    assert result["priority"] == "P2"
    assert result["category"] == "account_access"
    assert "password-reset.md" in result["sources"]
