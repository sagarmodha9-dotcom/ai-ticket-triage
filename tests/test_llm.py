import boto3
from botocore.stub import Stubber

from llm import BedrockLLM


def test_bedrock_converse_request_is_valid():
    """Botocore's Stubber checks our request against the real Bedrock API definition."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="x")
    stubber = Stubber(client)
    stubber.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": '{"priority": "P2"}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "metrics": {"latencyMs": 100},
        },
        {
            "modelId": "test-model",
            "system": [{"text": "sys"}],
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 800, "temperature": 0},
        },
    )
    with stubber:
        assert BedrockLLM("test-model", client=client).complete("sys", "hi") == '{"priority": "P2"}'


import io
import json

import pytest

from llm import AnthropicLLM, FakeLLM, FallbackLLM, build_llm


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_anthropic_request_shape_and_reply():
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = {k.lower(): v for k, v in request.header_items()}
        seen["body"] = json.loads(request.data)
        return _FakeResponse(json.dumps({"content": [{"type": "text", "text": '{"priority": "P1"}'}]}).encode())

    llm = AnthropicLLM("claude-haiku-4-5", "test-key", opener=opener)
    assert llm.complete("sys", "hi") == '{"priority": "P1"}'
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-haiku-4-5"
    assert seen["body"]["system"] == "sys"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_requires_key():
    with pytest.raises(ValueError):
        AnthropicLLM("claude-haiku-4-5", "")


class _Broken:
    provider = "bedrock"

    def complete(self, system, user, max_tokens=800):
        raise RuntimeError("ThrottlingException")


def test_fallback_uses_backup_when_primary_fails():
    llm = FallbackLLM(_Broken(), FakeLLM())
    assert json.loads(llm.complete("sys", "VPN is down"))["priority"]
    assert llm.provider == "fake"


def test_fallback_prefers_primary_when_it_works():
    llm = FallbackLLM(FakeLLM(), _Broken())
    llm.complete("sys", "printer jam")
    assert llm.provider == "fake"


def test_build_llm_rejects_unknown_provider():
    with pytest.raises(ValueError):
        build_llm("openai")
