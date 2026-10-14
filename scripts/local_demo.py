"""Triage one ticket on your own computer and print the result.

  python scripts/local_demo.py "VPN won't connect" "Can't reach internal sites from home"
  python scripts/local_demo.py --bedrock MODEL_ID "subject" "description"
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from knowledge import KnowledgeBase, load_from_dir  # noqa: E402
from llm import BedrockLLM, FakeLLM  # noqa: E402
from redact import redact  # noqa: E402
from triage import triage_ticket  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("subject")
parser.add_argument("description")
parser.add_argument("--bedrock", metavar="MODEL_ID")
args = parser.parse_args()

subject, c1 = redact(args.subject)
description, c2 = redact(args.description)
llm = BedrockLLM(args.bedrock) if args.bedrock else FakeLLM()
kb = KnowledgeBase(load_from_dir(os.path.join(ROOT, "knowledge_base")))
result = triage_ticket(subject, description, llm, kb)
result["redacted"] = {**c1, **c2}
print(json.dumps(result, indent=2))
