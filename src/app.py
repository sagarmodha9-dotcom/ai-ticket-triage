"""AWS Lambda handlers, called by API Gateway.

POST /tickets              -> create_ticket: triage a new ticket and save it
GET  /tickets/{ticket_id}  -> get_ticket:    look up a saved ticket
POST /agent                -> agent:         ask the help desk agent a question (it uses tools)
"""
import base64
import json
import logging
import os
import time
import uuid
from decimal import Decimal

from agent import HelpDeskTools, run_agent
from knowledge import KnowledgeBase, load_from_dir, load_from_s3
from llm import build_llm
from redact import redact
from store import DynamoTicketStore
from triage import triage_ticket

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_SUBJECT = 200
MAX_DESCRIPTION = 5000
MAX_QUESTION = 2000

# Created once per Lambda container and reused on warm starts (faster, cheaper).
_clients = {}


def _get(name):
    if name not in _clients:
        import boto3
        if name == "table":
            _clients[name] = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
        elif name == "sns":
            _clients[name] = boto3.client("sns")
        elif name == "llm":
            provider = os.environ.get("LLM_PROVIDER", "bedrock")
            api_key = ""
            if provider != "bedrock":
                # The key lives encrypted in SSM Parameter Store, never in code or env vars.
                api_key = boto3.client("ssm").get_parameter(
                    Name=os.environ["ANTHROPIC_KEY_PARAM"], WithDecryption=True)["Parameter"]["Value"]
            _clients[name] = build_llm(provider, os.environ.get("MODEL_ID", ""),
                                       os.environ.get("ANTHROPIC_MODEL_ID", ""), api_key)
        elif name == "kb":
            local_dir = os.environ.get("KB_LOCAL_DIR")
            if local_dir:
                articles = load_from_dir(local_dir)
            else:
                articles = load_from_s3(os.environ["KB_BUCKET"], os.environ.get("KB_PREFIX", "kb/"),
                                        boto3.client("s3"))
            _clients[name] = KnowledgeBase(articles)
        elif name == "store":
            _clients[name] = DynamoTicketStore(_get("table"))
    return _clients[name]


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=_json_default),
    }


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value) if value % 1 else int(value)
    raise TypeError(f"Can't serialize {type(value)}")


def _parse_body(event: dict) -> dict:
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("body must be a JSON object")
    return data


def create_ticket(event, context):
    try:
        data = _parse_body(event)
    except (ValueError, json.JSONDecodeError):
        return _response(400, {"error": "Request body must be valid JSON."})

    subject = str(data.get("subject", "")).strip()
    description = str(data.get("description", "")).strip()
    requester = str(data.get("requester", "")).strip()
    if not subject or not description:
        return _response(400, {"error": "Both 'subject' and 'description' are required."})
    if len(subject) > MAX_SUBJECT or len(description) > MAX_DESCRIPTION:
        return _response(400, {"error": f"Subject max {MAX_SUBJECT} chars, description max {MAX_DESCRIPTION}."})

    ticket_id = str(uuid.uuid4())
    clean_subject, c1 = redact(subject)
    clean_description, c2 = redact(description)
    redacted = {k: c1.get(k, 0) + c2.get(k, 0) for k in set(c1) | set(c2)}

    started = time.time()
    model_provider = "none"
    try:
        llm = _get("llm")
        result = triage_ticket(clean_subject, clean_description, llm, _get("kb"))
        model_provider = getattr(llm, "provider", "unknown")
    except Exception:
        # If the AI call fails (throttling, outage), still save the ticket for a human.
        logger.exception("Triage failed for %s", ticket_id)
        result = {
            "priority": "P3", "category": "other", "summary": "",
            "suggested_reply": "Thanks for contacting the help desk. A technician will follow up shortly.",
            "confidence": 0.0, "sources": [], "needs_human_review": True,
            "review_reasons": ["AI triage failed"],
        }
    latency_ms = int((time.time() - started) * 1000)

    item = {
        "ticket_id": ticket_id,
        "created_at": int(time.time()),
        "subject": subject,
        "description": description,
        "requester": requester,
        "status": "open",
        "redacted_counts": redacted,
        "model_provider": model_provider,
        **result,
        "confidence": Decimal(str(result["confidence"])),
    }
    _get("table").put_item(Item=item)

    if result["priority"] == "P1":
        _get("sns").publish(
            TopicArn=os.environ["ALERT_TOPIC_ARN"],
            Subject=f"P1 ticket: {clean_subject[:80]}",
            Message=f"Ticket {ticket_id}\nCategory: {result['category']}\nSummary: {result['summary']}",
        )

    # One structured log line per ticket, easy to search in CloudWatch Logs Insights.
    logger.info(json.dumps({"event": "ticket_triaged", "ticket_id": ticket_id,
                            "priority": result["priority"], "category": result["category"],
                            "needs_human_review": result["needs_human_review"],
                            "model_provider": model_provider,
                            "latency_ms": latency_ms, "redacted": redacted}))
    return _response(201, {k: v for k, v in item.items() if k != "description"})


def _p1_alert(ticket_id: str, reason: str) -> None:
    _get("sns").publish(
        TopicArn=os.environ["ALERT_TOPIC_ARN"],
        Subject="P1 escalated by help desk agent",
        Message=f"Ticket {ticket_id} was escalated to P1.\nReason: {reason}",
    )


def agent(event, context):
    try:
        data = _parse_body(event)
    except (ValueError, json.JSONDecodeError):
        return _response(400, {"error": "Request body must be valid JSON."})

    question = str(data.get("message", "")).strip()
    ticket_id = str(data.get("ticket_id") or "").strip() or None
    if not question:
        return _response(400, {"error": "'message' is required."})
    if len(question) > MAX_QUESTION:
        return _response(400, {"error": f"Message max {MAX_QUESTION} chars."})
    if ticket_id:
        try:
            ticket_id = str(uuid.UUID(ticket_id))
        except ValueError:
            return _response(400, {"error": "Invalid ticket_id."})

    started = time.time()
    llm = _get("llm")
    tools = HelpDeskTools(_get("store"), _get("kb"), notifier=_p1_alert)
    try:
        result = run_agent(question, llm, tools, ticket_id=ticket_id)
    except Exception:
        # Every model provider failed. Report it plainly; nothing was half-changed silently.
        logger.exception("Agent failed")
        return _response(503, {"error": "The AI model is unavailable right now. Try again shortly.",
                               "actions": tools.actions})
    result["latency_ms"] = int((time.time() - started) * 1000)

    logger.info(json.dumps({"event": "agent_run", "iterations": result["iterations"],
                            "stopped": result["stopped"], "tools": [s["tool"] for s in result["steps"]],
                            "actions": [a["action"] for a in result["actions"]],
                            "model_provider": result["model_provider"], "usage": result["usage"],
                            "latency_ms": result["latency_ms"]}))
    return _response(200, result)


def get_ticket(event, context):
    ticket_id = (event.get("pathParameters") or {}).get("ticket_id", "")
    try:
        uuid.UUID(ticket_id)
    except ValueError:
        return _response(400, {"error": "Invalid ticket_id."})
    item = _get("table").get_item(Key={"ticket_id": ticket_id}).get("Item")
    if not item:
        return _response(404, {"error": "Ticket not found."})
    return _response(200, item)
