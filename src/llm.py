"""Talk to the AI model.

BedrockLLM calls a Claude model through Amazon Bedrock's Converse API.
FakeLLM is a simple keyword-based stand-in so tests and demos run without AWS.
Both have the same complete(system, user) method, so the rest of the code
doesn't care which one it is using.
"""
import json
import re


class BedrockLLM:
    def __init__(self, model_id: str, client=None):
        self.model_id = model_id
        if client is None:
            import boto3  # only needed when actually calling AWS
            client = boto3.client("bedrock-runtime")
        self.client = client

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        return response["output"]["message"]["content"][0]["text"]


class FakeLLM:
    """Pretends to be the AI using keyword rules. Good enough to test the pipeline."""

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
