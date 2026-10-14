"""Score the triage pipeline against tickets with known answers.

Usage:
  python evals/run_evals.py                 # uses the fake model (no AWS needed)
  python evals/run_evals.py --bedrock MODEL_ID   # uses the real model in Amazon Bedrock
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedrock", metavar="MODEL_ID", help="call the real model in Bedrock")
    args = parser.parse_args()

    llm = BedrockLLM(args.bedrock) if args.bedrock else FakeLLM()
    kb = KnowledgeBase(load_from_dir(os.path.join(ROOT, "knowledge_base")))

    with open(os.path.join(ROOT, "evals", "tickets.jsonl")) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    priority_ok = category_ok = 0
    for case in cases:
        subject, _ = redact(case["subject"])
        description, _ = redact(case["description"])
        result = triage_ticket(subject, description, llm, kb)
        p = result["priority"] == case["expected_priority"]
        c = result["category"] == case["expected_category"]
        priority_ok += p
        category_ok += c
        mark = "PASS" if p and c else "FAIL"
        print(f"{mark}  {case['subject'][:35]:35}  got {result['priority']}/{result['category']:20}"
              f"  expected {case['expected_priority']}/{case['expected_category']}")

    n = len(cases)
    print(f"\nPriority accuracy: {priority_ok}/{n} ({priority_ok / n:.0%})")
    print(f"Category accuracy: {category_ok}/{n} ({category_ok / n:.0%})")


if __name__ == "__main__":
    main()
