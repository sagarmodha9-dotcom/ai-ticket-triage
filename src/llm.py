"""Talk to the AI model.

BedrockLLM    calls a model through Amazon Bedrock's Converse API.
AnthropicLLM  calls Claude directly through the Anthropic Messages API.
FallbackLLM   tries one provider and switches to a backup if it fails
              (throttling, quota, outage), so tickets still get triaged.
FakeLLM       is a simple keyword-based stand-in so tests and demos run without AWS.

All of them have the same complete(system, user) method, so the rest of the code
doesn't care which one it is using.
"""
import json
import logging
import re
import urllib.request

logger = logging.getLogger(__name__)


class BedrockLLM:
    provider = "bedrock"

    def __init__(self, model_id: str, client=None, max_attempts: int | None = None):
        self.model_id = model_id
        if client is None:
            import boto3  # only needed when actually calling AWS
            config = None
            if max_attempts is not None:
                from botocore.config import Config
                config = Config(retries={"max_attempts": max_attempts, "mode": "standard"})
            client = boto3.client("bedrock-runtime", config=config)
        self.client = client

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        return response["output"]["message"]["content"][0]["text"]


class AnthropicLLM:
    """Calls the Anthropic Messages API with only the standard library (no extra packages)."""

    provider = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model_id: str, api_key: str, opener=urllib.request.urlopen, timeout: int = 20):
        if not api_key:
            raise ValueError("Anthropic API key is missing")
        self.model_id = model_id
        self.api_key = api_key
        self.opener = opener
        self.timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        payload = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
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
            data = json.loads(response.read().decode("utf-8"))
        return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")


class FallbackLLM:
    """Try the primary provider first; if it raises, use the backup instead."""

    def __init__(self, primary, backup):
        self.primary = primary
        self.backup = backup
        self.last_provider = None

    @property
    def provider(self):
        return self.last_provider or self.primary.provider

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        try:
            text = self.primary.complete(system, user, max_tokens)
            self.last_provider = self.primary.provider
            return text
        except Exception as error:  # throttling, no quota, outage, etc.
            logger.warning("Primary model (%s) failed, switching to backup (%s): %s",
                           self.primary.provider, self.backup.provider, type(error).__name__)
            text = self.backup.complete(system, user, max_tokens)
            self.last_provider = self.backup.provider
            return text


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

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
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


def build_llm(provider: str, bedrock_model_id: str = "", anthropic_model_id: str = "",
              anthropic_api_key: str = ""):
    """Pick the model setup by name: bedrock, anthropic, or bedrock_with_anthropic_fallback."""
    if provider == "bedrock":
        return BedrockLLM(bedrock_model_id)
    if provider == "anthropic":
        return AnthropicLLM(anthropic_model_id, anthropic_api_key)
    if provider == "bedrock_with_anthropic_fallback":
        # One quick Bedrock attempt (no long retry loop), then fail over to Anthropic.
        return FallbackLLM(BedrockLLM(bedrock_model_id, max_attempts=1),
                           AnthropicLLM(anthropic_model_id, anthropic_api_key))
    raise ValueError(f"Unknown LLM provider: {provider}")
