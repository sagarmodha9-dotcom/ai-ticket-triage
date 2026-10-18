"""Score the triage pipeline against tickets with known answers.

Usage:
  python evals/run_evals.py                 # uses the fake model (no AWS needed)
  python evals/run_evals.py --bedrock MODEL_ID   # uses the real model in Amazon Bedrock
  python evals/run_evals.py --anthropic claude-haiku-4-5   # uses Claude via the Anthropic API
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from knowledge import KnowledgeBase, load_from_dir  # noqa: E402
from llm import AnthropicLLM, BedrockLLM, FakeLLM  # noqa: E402
from redact import redact  # noqa: E402
from triage import triage_ticket  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bedrock", metavar="MODEL_ID", help="call the real model in Bedrock")
    parser.add_argument("--anthropic", metavar="MODEL_ID", help="call Claude via the Anthropic API (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--min-accuracy", type=float, default=0.0,
                        help="exit with an error if priority or category accuracy is below this (0-1)")
    args = parser.parse_args()

    if args.anthropic:
        llm = AnthropicLLM(args.anthropic, os.environ.get("ANTHROPIC_API_KEY", ""))
    elif args.bedrock:
        llm = BedrockLLM(args.bedrock)
    else:
        llm = FakeLLM()

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
    if min(priority_ok, category_ok) / n < args.min_accuracy:
        print(f"FAILED: accuracy below the {args.min_accuracy:.0%} minimum")
        sys.exit(1)


if __name__ == "__main__":
    main()
