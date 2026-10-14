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
