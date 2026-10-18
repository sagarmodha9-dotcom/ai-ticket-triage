"""Talk to the AI model.

BedrockLLM    calls a model through Amazon Bedrock's Converse API.
AnthropicLLM  calls Claude directly through the Anthropic Messages API.
FallbackLLM   tries one provider and switches to a backup if it fails
              (throttling, quota, outage), so tickets still get triaged.
FakeLLM       is a simple keyword-based stand-in so tests and demos run without AWS.

All of them have the same two methods, so the rest of the code doesn't care which
one it is using:

  complete(system, user)        one prompt in, text out (used by ticket triage)
  chat(system, messages, tools) one turn of a tool-using conversation (used by the agent)

Messages for chat() use one neutral format, converted to each provider's format here:
  {"role": "user" | "assistant", "content": [block, ...]}
  block = {"type": "text", "text": ...}
        | {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
        | {"type": "tool_result", "tool_use_id": ..., "content": "...", "is_error": bool}
chat() returns {"content": [blocks], "stop_reason": "tool_use" | "end_turn" | ...,
                "usage": {"input_tokens": n, "output_tokens": n}}
"""
import json
import logging
import re
import urllib.request

logger = logging.getLogger(__name__)


def _usage(input_tokens=0, output_tokens=0) -> dict:
    return {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0)}


class BedrockLLM:
    provider = "bedrock"

    def __init__(self, model_id: str, client=None, max_attempts: int | None = None, read_timeout: int = 15):
        self.model_id = model_id
        if client is None:
            import boto3  # only needed when actually calling AWS
            from botocore.config import Config
            # botocore's default read timeout is 60s, longer than API Gateway's 30s limit.
            config = Config(connect_timeout=3, read_timeout=read_timeout,
                            retries={"max_attempts": max_attempts or 3, "mode": "standard"})
            client = boto3.client("bedrock-runtime", config=config)
        self.client = client

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = response.get("usage", {})
        self.last_usage = _usage(usage.get("inputTokens"), usage.get("outputTokens"))
        return response["output"]["message"]["content"][0]["text"]

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 800) -> dict:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=to_bedrock_messages(messages),
            toolConfig={"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                                "inputSchema": {"json": t["input_schema"]}}}
                                  for t in tools]},
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        content = []
        for block in response["output"]["message"]["content"]:
            if "text" in block:
                content.append({"type": "text", "text": block["text"]})
            elif "toolUse" in block:
                use = block["toolUse"]
                content.append({"type": "tool_use", "id": use["toolUseId"], "name": use["name"],
                                "input": use.get("input") or {}})
        usage = response.get("usage", {})
        return {"content": content, "stop_reason": response.get("stopReason", ""),
                "usage": _usage(usage.get("inputTokens"), usage.get("outputTokens"))}


def to_bedrock_messages(messages: list) -> list:
    """Neutral message format -> Bedrock Converse format."""
    out = []
    for message in messages:
        blocks = []
        for b in message["content"]:
            if b["type"] == "text" and b["text"].strip():
                blocks.append({"text": b["text"]})
            elif b["type"] == "tool_use":
                blocks.append({"toolUse": {"toolUseId": b["id"], "name": b["name"], "input": b["input"]}})
            elif b["type"] == "tool_result":
                blocks.append({"toolResult": {"toolUseId": b["tool_use_id"],
                                              "content": [{"text": b["content"]}],
                                              "status": "error" if b.get("is_error") else "success"}})
        out.append({"role": message["role"], "content": blocks})
    return out


class AnthropicLLM:
    """Calls the Anthropic Messages API with only the standard library (no extra packages)."""

    provider = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model_id: str, api_key: str, opener=urllib.request.urlopen, timeout: int = 15):
        if not api_key:
            raise ValueError("Anthropic API key is missing")
        self.model_id = model_id
        self.api_key = api_key
        self.opener = opener
        self.timeout = timeout

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with self.opener(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        data = self._post({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        })
        usage = data.get("usage", {})
        self.last_usage = _usage(usage.get("input_tokens"), usage.get("output_tokens"))
        return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 800) -> dict:
        data = self._post({
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "tools": tools,
        })
        content = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content.append({"type": "text", "text": block.get("text", "")})
            elif block.get("type") == "tool_use":
                content.append({"type": "tool_use", "id": block["id"], "name": block["name"],
                                "input": block.get("input") or {}})
        usage = data.get("usage", {})
        return {"content": content, "stop_reason": data.get("stop_reason", ""),
                "usage": _usage(usage.get("input_tokens"), usage.get("output_tokens"))}


def to_anthropic_messages(messages: list) -> list:
    """Neutral message format -> Anthropic Messages format (nearly the same; drop empty text)."""
    out = []
    for message in messages:
        blocks = []
        for b in message["content"]:
            if b["type"] == "text":
                if b["text"].strip():
                    blocks.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                blocks.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]})
            elif b["type"] == "tool_result":
                blocks.append({"type": "tool_result", "tool_use_id": b["tool_use_id"],
                               "content": b["content"], "is_error": bool(b.get("is_error"))})
        out.append({"role": message["role"], "content": blocks})
    return out


class FallbackLLM:
    """Try the primary provider first; if it raises, use the backup instead."""

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup = backup
        self.last_provider = None

    @property
    def provider(self):
        return self.last_provider or self.primary.provider

    @property
    def last_usage(self):
        used = self.backup if self.last_provider == self.backup.provider else self.primary
        return getattr(used, "last_usage", _usage())

    def _call(self, method: str, *args):
        try:
            result = getattr(self.primary, method)(*args)
            self.last_provider = self.primary.provider
            return result
        except Exception as error:  # throttling, no quota, outage, etc.
            logger.warning("Primary model (%s) failed, switching to backup (%s): %s",
                           self.primary.provider, self.backup.provider, type(error).__name__)
            result = getattr(self.backup, method)(*args)
            self.last_provider = self.backup.provider
            return result

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        return self._call("complete", system, user, max_tokens)

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 800) -> dict:
        return self._call("chat", system, messages, tools, max_tokens)


class FakeLLM:
    """Pretends to be the AI using keyword rules. Good enough to test the pipeline."""

    provider = "fake"

    RULES = [
        # (keywords, priority, category)
        (("phishing", "ransom", "bitcoin", "malware", "virus", "suspicious", "hacked", "breach"), "P1", "security"),
        (("everyone", "entire", "whole office", "all users", "outage", "down for"), "P1", "network"),
        (("vpn",), "P2", "network"),
        (("wifi", "wi-fi", "internet", "network"), "P2", "network"),
        (("locked out", "password", "mfa", "authenticator", "sign in", "login", "log in"), "P2", "account_access"),
        (("outlook", "email", "mailbox", "calendar", "teams"), "P3", "email_collaboration"),
        (("printer", "print", "monitor", "keyboard", "laptop", "screen", "battery", "slow"), "P3", "hardware"),
        (("install", "license", "software", "adobe", "access to", "request"), "P4", "software_request"),
    ]
    TOPICS = ("vpn", "wifi", "internet", "outlook", "email", "teams", "printer", "password",
              "mfa", "phishing", "laptop", "network")
    UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        self.last_usage = _usage(len(system + user) // 4, 60)
        ticket = user.split("ARTICLES:")[0].lower()
        priority, category = "P3", "other"
        for keywords, p, c in self.RULES:
            if any(k in ticket for k in keywords):
                priority, category = p, c
                break
        sources = re.findall(r"\[source: ([^\]]+)\]", user)
        return json.dumps({
            "priority": priority,
            "category": category,
            "summary": "Fake model summary of the ticket.",
            "suggested_reply": "Thanks for reaching out. We're looking into this now.",
            "confidence": 0.8,
            "sources": sources[:2],
        })

    def chat(self, system: str, messages: list, tools: list, max_tokens: int = 800) -> dict:
        """A scripted 'agent': look things up first, escalate if there's a pattern, then answer."""
        question = " ".join(b["text"] for b in messages[0]["content"] if b["type"] == "text").lower()
        results = {}
        for message in messages:
            for b in message["content"]:
                if b["type"] == "tool_use":
                    results[b["id"]] = {"name": b["name"]}
                elif b["type"] == "tool_result" and b["tool_use_id"] in results:
                    try:
                        results[b["tool_use_id"]]["output"] = json.loads(b["content"])
                    except json.JSONDecodeError:
                        results[b["tool_use_id"]]["output"] = {}
        done = {r["name"]: r.get("output", {}) for r in results.values()}
        step = len(results)

        def call(name, args):
            return {"content": [{"type": "tool_use", "id": f"fake_{step}_{name}", "name": name, "input": args}],
                    "stop_reason": "tool_use", "usage": _usage(100, 20)}

        topic = next((t for t in self.TOPICS if t in question), None)
        ticket_match = self.UUID.search(question)
        if ticket_match and "get_ticket" not in done:
            return call("get_ticket", {"ticket_id": ticket_match.group(0)})
        if not topic and "get_ticket" in done:
            subject = str(done["get_ticket"].get("untrusted_ticket_text", {}).get("subject", "")).lower()
            topic = next((t for t in self.TOPICS if t in subject), None)
        if topic and "search_recent_tickets" not in done:
            return call("search_recent_tickets", {"keyword": topic, "hours": 24})
        if topic and "search_knowledge_base" not in done:
            return call("search_knowledge_base", {"query": topic})
        similar = done.get("search_recent_tickets", {}).get("tickets", [])
        if len(similar) >= 3 and "escalate_ticket" not in done:
            newest = max(similar, key=lambda t: t.get("created_at", 0))
            return call("escalate_ticket", {"ticket_id": newest["ticket_id"], "priority": "P1",
                                            "reason": f"Possible outage: {len(similar)} similar {topic} tickets in 24 hours"})
        if similar:
            answer = (f"I found {len(similar)} recent {topic} ticket(s). "
                      + ("That looks like an outage, so I escalated the newest one to P1. "
                         if "escalate_ticket" in done else "No pattern big enough to call an outage. ")
                      + "Next step: check the shared service and follow the knowledge base article.")
        else:
            answer = "I didn't find related tickets. Next step: follow the knowledge base article or assign a technician."
        return {"content": [{"type": "text", "text": answer}], "stop_reason": "end_turn", "usage": _usage(150, 60)}


def build_llm(provider: str, bedrock_model_id: str = "", anthropic_model_id: str = "",
              anthropic_api_key: str = ""):
    """Pick the model setup by name: bedrock, anthropic, bedrock_with_anthropic_fallback, or fake."""
    if provider == "bedrock":
        return BedrockLLM(bedrock_model_id)
    if provider == "anthropic":
        return AnthropicLLM(anthropic_model_id, anthropic_api_key)
    if provider == "bedrock_with_anthropic_fallback":
        # One quick Bedrock attempt (no long retry loop), then fail over to Anthropic.
        return FallbackLLM(BedrockLLM(bedrock_model_id, max_attempts=1),
                           AnthropicLLM(anthropic_model_id, anthropic_api_key))
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown LLM provider: {provider}")
