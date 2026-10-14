"""The core triage logic: build the prompt, ask the AI, check its answer.

Flow for one ticket:
  1. Hide personal data (redact.py)
  2. Find the most relevant help desk articles (knowledge.py)
  3. Ask the AI for priority, category, summary and a suggested reply (llm.py)
  4. Validate the answer and apply safety rules the AI can't override
"""
import json
import re

PRIORITIES = {
    "P1": "Critical: security incident, or many users / a whole site can't work.",
    "P2": "High: one user fully blocked from working (locked out, no VPN, no network).",
    "P3": "Medium: degraded but can still work (slow laptop, printer, email glitch).",
    "P4": "Low: requests and questions (new software, access requests, how-to).",
}
CATEGORIES = ["account_access", "network", "email_collaboration", "hardware",
              "software_request", "security", "other"]

# Words that mean "possible security incident". If any appear, the ticket is at
# least P2 and always goes to a human, no matter what the AI says.
SECURITY_WORDS = ("phishing", "ransom", "bitcoin", "malware", "virus", "hacked", "breach",
                  "suspicious link", "clicked a link", "stolen", "compromised")

CONFIDENCE_THRESHOLD = 0.6

SYSTEM_PROMPT = """You are an IT help desk triage assistant.
Read the ticket and the help desk articles, then respond with ONLY a JSON object:
{
  "priority": one of P1, P2, P3, P4,
  "category": one of %(categories)s,
  "summary": one sentence describing the problem,
  "suggested_reply": a short, friendly reply to the user with next steps,
  "confidence": a number from 0 to 1 for how sure you are,
  "sources": list of article file names you used (empty if none helped)
}
Priority definitions:
%(priorities)s
Rules:
- Base the suggested reply on the articles when they apply. Do not invent steps.
- If the articles don't cover the issue, say a technician will follow up.
- Never ask the user for their password.
- Output JSON only, no extra text."""


def build_system_prompt() -> str:
    priorities = "\n".join(f"- {k}: {v}" for k, v in PRIORITIES.items())
    return SYSTEM_PROMPT % {"categories": ", ".join(CATEGORIES), "priorities": priorities}


def build_user_message(subject: str, description: str, articles) -> str:
    parts = [f"TICKET SUBJECT: {subject}", f"TICKET DESCRIPTION: {description}", "", "ARTICLES:"]
    if not articles:
        parts.append("(no relevant articles found)")
    for a in articles:
        parts.append(f"[source: {a.source}]\n{a.text.strip()}\n")
    return "\n".join(parts)


def parse_model_output(text: str) -> dict | None:
    """Pull the JSON object out of the model's reply. Returns None if it's unusable."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate(data: dict | None) -> tuple[dict, list[str]]:
    """Make sure every field is present and allowed. Returns (clean_result, problems)."""
    problems = []
    if data is None:
        data, problems = {}, ["model output was not valid JSON"]

    priority = str(data.get("priority", "")).upper()
    if priority not in PRIORITIES:
        problems.append(f"invalid priority {priority!r}")
        priority = "P3"

    category = str(data.get("category", "")).lower()
    if category not in CATEGORIES:
        problems.append(f"invalid category {category!r}")
        category = "other"

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        problems.append("invalid confidence")
        confidence = 0.0

    sources = data.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    result = {
        "priority": priority,
        "category": category,
        "summary": str(data.get("summary") or "").strip()[:500],
        "suggested_reply": str(data.get("suggested_reply") or "").strip()[:2000]
        or "Thanks for contacting the help desk. A technician will follow up shortly.",
        "confidence": round(confidence, 2),
        "sources": [str(s) for s in sources][:5],
    }
    return result, problems


def apply_guardrails(result: dict, ticket_text: str, problems: list[str]) -> dict:
    """Rules in plain code that the AI can't talk its way around."""
    reasons = []
    text = ticket_text.lower()
    if any(word in text for word in SECURITY_WORDS):
        if result["priority"] in ("P3", "P4"):
            result["priority"] = "P2"
            reasons.append("security keywords: raised priority to P2")
        result["category"] = "security"
        reasons.append("possible security incident")
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        reasons.append(f"low confidence ({result['confidence']})")
    if problems:
        reasons.extend(problems)
    result["needs_human_review"] = bool(reasons)
    result["review_reasons"] = reasons
    return result


def triage_ticket(subject: str, description: str, llm, kb) -> dict:
    """Run one ticket through the whole pipeline. Text passed in should already be redacted."""
    articles = kb.search(f"{subject} {description}", top_k=3)
    raw = llm.complete(build_system_prompt(), build_user_message(subject, description, articles))
    result, problems = validate(parse_model_output(raw))
    # Only keep sources that were really given to the model.
    allowed = {a.source for a in articles}
    result["sources"] = [s for s in result["sources"] if s in allowed]
    return apply_guardrails(result, f"{subject} {description}", problems)
