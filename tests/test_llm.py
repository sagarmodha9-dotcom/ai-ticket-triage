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


# ---- tool calling (used by the agent) ---------------------------------------------------

from botocore.stub import ANY  # noqa: E402

from agent import TOOLS  # noqa: E402
from llm import to_anthropic_messages, to_bedrock_messages  # noqa: E402

CONVERSATION = [
    {"role": "user", "content": [{"type": "text", "text": "Any VPN outage?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": ""},
                                      {"type": "tool_use", "id": "tu_1", "name": "search_recent_tickets",
                                       "input": {"keyword": "vpn"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1",
                                  "content": '{"count": 3}', "is_error": False}]},
]


def test_bedrock_tool_request_is_valid_and_reply_is_parsed():
    """Stubber checks toolConfig, toolUse and toolResult against the real Bedrock API definition."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="x")
    stubber = Stubber(client)
    stubber.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [
                {"text": "Escalating."},
                {"toolUse": {"toolUseId": "tu_2", "name": "escalate_ticket",
                             "input": {"ticket_id": "x", "reason": "outage"}}}]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 120, "outputTokens": 30, "totalTokens": 150},
            "metrics": {"latencyMs": 100},
        },
        {"modelId": "test-model", "system": [{"text": "sys"}], "messages": ANY,
         "toolConfig": ANY, "inferenceConfig": {"maxTokens": 800, "temperature": 0}},
    )
    with stubber:
        reply = BedrockLLM("test-model", client=client).chat("sys", CONVERSATION, TOOLS)
    assert reply["stop_reason"] == "tool_use"
    assert reply["content"][1] == {"type": "tool_use", "id": "tu_2", "name": "escalate_ticket",
                                   "input": {"ticket_id": "x", "reason": "outage"}}
    assert reply["usage"] == {"input_tokens": 120, "output_tokens": 30}


def test_bedrock_message_conversion():
    converted = to_bedrock_messages(CONVERSATION)
    assert converted[1]["content"] == [{"toolUse": {"toolUseId": "tu_1", "name": "search_recent_tickets",
                                                     "input": {"keyword": "vpn"}}}]  # empty text dropped
    assert converted[2]["content"][0]["toolResult"]["status"] == "success"


def test_bedrock_full_request_passes_api_validation():
    """Same as above but without ANY: botocore validates every field we send."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="x", aws_secret_access_key="x")
    stubber = Stubber(client)
    expected = {
        "modelId": "m", "system": [{"text": "sys"}], "messages": to_bedrock_messages(CONVERSATION),
        "toolConfig": {"tools": [{"toolSpec": {"name": t["name"], "description": t["description"],
                                               "inputSchema": {"json": t["input_schema"]}}} for t in TOOLS]},
        "inferenceConfig": {"maxTokens": 800, "temperature": 0},
    }
    stubber.add_response("converse", {
        "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
        "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1}}, expected)
    with stubber:
        assert BedrockLLM("m", client=client).chat("sys", CONVERSATION, TOOLS)["content"] == [
            {"type": "text", "text": "done"}]


def test_anthropic_tool_request_and_reply():
    seen = {}

    def opener(request, timeout):
        seen["payload"] = json.loads(request.data)
        return _FakeResponse(json.dumps({
            "content": [{"type": "tool_use", "id": "toolu_9", "name": "get_ticket",
                         "input": {"ticket_id": "abc"}}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 12},
        }).encode())

    reply = AnthropicLLM("claude-haiku-4-5", "key", opener=opener).chat("sys", CONVERSATION, TOOLS)
    assert seen["payload"]["tools"] == TOOLS
    assert seen["payload"]["messages"] == to_anthropic_messages(CONVERSATION)
    assert seen["payload"]["messages"][1]["content"][0]["type"] == "tool_use"  # empty text dropped
    assert reply["content"][0]["name"] == "get_ticket" and reply["usage"]["output_tokens"] == 12


def test_fallback_chat_switches_provider():
    class Down:
        provider = "bedrock"

        def chat(self, *a):
            raise RuntimeError("ThrottlingException")

    llm = FallbackLLM(Down(), FakeLLM())
    reply = llm.chat("sys", CONVERSATION, TOOLS)
    assert llm.provider == "fake" and reply["content"]
