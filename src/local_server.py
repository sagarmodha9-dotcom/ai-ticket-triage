"""Run the whole API on your own computer (or in Docker) with no AWS account.

Same handlers as AWS Lambda, but tickets are kept in memory and P1 alerts are
printed instead of emailed.

  python src/local_server.py                 # fake model, no keys needed
  LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... python src/local_server.py   # real Claude

Routes: POST /tickets, GET /tickets/{id}, POST /agent, GET /health
"""
import json
import logging
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app  # noqa: E402
from knowledge import KnowledgeBase, load_from_dir  # noqa: E402
from llm import build_llm  # noqa: E402
from store import MemoryTable, MemoryTicketStore  # noqa: E402

DEFAULT_KB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")


class PrintSNS:
    """Stands in for Amazon SNS: prints the alert instead of emailing it."""

    def publish(self, **kwargs):
        print(f"\n*** ALERT: {kwargs.get('Subject')}\n{kwargs.get('Message')}\n", flush=True)


DEMO_TICKETS = [
    ("VPN won't connect", "VPN keeps failing since this morning, error 809.", 50),
    ("VPN down?", "Can't get on the VPN from home. Tried restarting twice.", 35),
    ("vpn disconnecting", "VPN drops every few minutes, can't reach the file share.", 20),
    ("Printer jam on 3rd floor", "The HP printer by the kitchen keeps jamming.", 90),
]


def seed_demo_tickets(table: MemoryTable) -> None:
    """A few recent tickets so the agent has something to find (3 VPN tickets = possible outage)."""
    now = int(time.time())
    for subject, description, minutes_ago in DEMO_TICKETS:
        ticket_id = str(uuid.uuid4())
        table.put_item(Item={
            "ticket_id": ticket_id, "created_at": now - minutes_ago * 60, "subject": subject,
            "description": description, "requester": "demo@example.com", "status": "open",
            "priority": "P2" if "vpn" in subject.lower() else "P3",
            "category": "network" if "vpn" in subject.lower() else "hardware",
            "needs_human_review": False, "review_reasons": [], "model_provider": "seed",
        })


def configure(provider: str | None = None, seed: bool | None = None) -> MemoryTable:
    """Wire app.py to in-memory storage and the chosen model. Returns the table (handy for tests)."""
    provider = provider or os.environ.get("LLM_PROVIDER", "fake")
    if seed is None:
        seed = os.environ.get("SEED_DEMO_TICKETS", "true").lower() == "true"
    os.environ.setdefault("ALERT_TOPIC_ARN", "local-alerts")
    table = MemoryTable()
    if seed:
        seed_demo_tickets(table)
    app._clients.clear()
    app._clients.update({
        "table": table,
        "store": MemoryTicketStore(table),
        "sns": PrintSNS(),
        "kb": KnowledgeBase(load_from_dir(os.environ.get("KB_LOCAL_DIR", DEFAULT_KB))),
        "llm": build_llm(provider, os.environ.get("MODEL_ID", ""),
                         os.environ.get("ANTHROPIC_MODEL_ID", "claude-haiku-4-5"),
                         os.environ.get("ANTHROPIC_API_KEY", "")),
    })
    return table


class Handler(BaseHTTPRequestHandler):
    def _send(self, response: dict) -> None:
        body = response["body"].encode("utf-8")
        self.send_response(response["statusCode"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> str:
        length = min(int(self.headers.get("Content-Length") or 0), 100_000)
        return self.rfile.read(length).decode("utf-8") if length else "{}"

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            return self._send({"statusCode": 200, "body": json.dumps({"status": "ok"})})
        if path.startswith("/tickets/"):
            ticket_id = path[len("/tickets/"):]
            return self._send(app.get_ticket({"pathParameters": {"ticket_id": ticket_id}}, None))
        if path == "/tickets":
            items = sorted(app._clients["table"].items.values(), key=lambda t: t["created_at"], reverse=True)
            summary = [{k: t.get(k) for k in ("ticket_id", "subject", "priority", "category", "status")}
                       for t in items]
            return self._send({"statusCode": 200, "body": json.dumps({"tickets": summary})})
        self._send({"statusCode": 404, "body": json.dumps({"error": "Not found"})})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        event = {"body": self._body(), "headers": dict(self.headers.items())}
        if path == "/tickets":
            return self._send(app.create_ticket(event, None))
        if path == "/agent":
            return self._send(app.agent(event, None))
        self._send({"statusCode": 404, "body": json.dumps({"error": "Not found"})})

    def log_message(self, fmt, *args):
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


def main():
    logging.basicConfig(level=logging.WARNING)
    configure()
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    provider = getattr(app._clients["llm"], "provider", "?")
    print(f"AI Ticket Triage running on http://localhost:{port}  (model: {provider})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
