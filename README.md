# AI Support Ticket Triage

A serverless help desk assistant on AWS. When a support ticket comes in, it:

1. **Hides personal data** (emails, phone numbers, IPs, SSNs) before anything reaches the AI
2. **Finds relevant help desk articles** in a knowledge base stored in S3 (retrieval / RAG)
3. **Asks Claude on Amazon Bedrock** to set a **priority (P1–P4)** and **category**, summarize the issue, and draft a reply based on the articles
4. **Checks the AI's answer** with plain-code guardrails: valid values only, no made-up sources, security tickets always escalated, low-confidence answers flagged for a human
5. **Saves the ticket** in DynamoDB and **emails an alert** through SNS when it's a P1

```
                 POST /tickets
 user/app ───────────────────────▶ API Gateway (HTTP API, throttled)
                                          │
                                          ▼
                                   Lambda: create_ticket
            ┌──────────────┬──────────────┼───────────────┬──────────────┐
            ▼              ▼              ▼               ▼              ▼
     redact PII     S3 knowledge     Amazon Bedrock    DynamoDB      SNS email
                    base search       (Claude)         (tickets)    (P1 only)

 GET /tickets/{id} ──▶ API Gateway ──▶ Lambda: get_ticket ──▶ DynamoDB
```

## Example

```bash
curl -X POST "$API_URL/tickets" -H "Content-Type: application/json" -d '{
  "subject": "VPN won't connect",
  "description": "Working from home and the VPN keeps failing. Call me at 301-555-0142.",
  "requester": "sam@example.com"
}'
```

```json
{
  "ticket_id": "6f1c...",
  "priority": "P2",
  "category": "network",
  "summary": "User can't connect to the VPN while working from home.",
  "suggested_reply": "Sorry about that! First, fully quit the VPN client and reopen it...",
  "confidence": 0.86,
  "sources": ["vpn.md"],
  "needs_human_review": false,
  "redacted_counts": {"phone": 1}
}
```

## Project layout

| Path | What it is |
|---|---|
| `template.yaml` | AWS SAM template: every AWS resource, defined as code |
| `src/app.py` | Lambda handlers for the two API routes |
| `src/triage.py` | Prompt, output validation, guardrails |
| `src/knowledge.py` | Loads articles from S3 and searches them (TF-IDF) |
| `src/redact.py` | Removes personal data before the AI call |
| `src/llm.py` | Bedrock Converse API client, Anthropic API client, provider failover, and a fake model for tests |
| `knowledge_base/` | Sample IT help desk articles |
| `tests/` | 21 unit tests (no AWS needed) |
| `evals/` | 12 tickets with known answers to measure accuracy |

## Run it locally (no AWS account needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                      # 21 tests
python evals/run_evals.py                # accuracy with the fake model
python scripts/local_demo.py "Locked out" "Too many password attempts"
```

## Deploy to AWS

Prerequisites: AWS account, [AWS CLI](https://aws.amazon.com/cli/) and [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) installed and configured.

1. In the **Amazon Bedrock console**, open the **Model catalog**, find the Claude model you want, and make sure you can use it in your Region. Copy its model ID or inference profile ID.
2. Deploy:
   ```bash
   sam build
   sam deploy --guided      # enter AlertEmail and ModelId when asked
   ```
3. Confirm the SNS subscription email AWS sends you.
4. Upload the knowledge base (bucket name is in the deploy outputs):
   ```bash
   ./scripts/upload_kb.sh <KnowledgeBucketName>
   ```
5. Test with the `curl` example above using the `ApiUrl` output.
6. Measure real accuracy: `python evals/run_evals.py --bedrock <MODEL_ID>` (or `--anthropic claude-haiku-4-5` with `ANTHROPIC_API_KEY` set)

**Optional: Anthropic API failover.** New AWS accounts can start with very low Bedrock quotas. To fail over to the Anthropic API, store your key once (it's read silently, so it never lands in your shell history):

```bash
read -s ANTHROPIC_KEY && aws ssm put-parameter --name /ai-ticket-triage/anthropic-api-key \
  --type SecureString --value "$ANTHROPIC_KEY" --region us-east-1 && unset ANTHROPIC_KEY
```

**Clean up** when finished: `aws s3 rm s3://<KnowledgeBucketName> --recursive && sam delete`

## Design decisions

- **Serverless** (Lambda, HTTP API, DynamoDB on-demand): no servers to manage and near-zero cost when idle.
- **Keyword (TF-IDF) retrieval instead of a vector database:** for a small knowledge base it's accurate enough, instant, and free. A managed vector store bills every hour it runs. Swapping in Bedrock Knowledge Bases is a natural next step at larger scale.
- **Guardrails in code, not just the prompt:** the model can be wrong, so security escalation, allowed values, and source checking are enforced by Python, and anything uncertain goes to a human.
- **PII redaction before the AI call:** private data stays inside the system.
- **Least-privilege IAM:** each function only gets the permissions it needs (the read-only function can't write).
- **API throttling:** caps requests so nobody can run up a Bedrock bill.
- **Provider failover:** by default the Lambda tries Claude on Bedrock first (one quick attempt, no long retry loop) and fails over to Claude through the Anthropic API if Bedrock is throttled or out of quota. Each ticket records which provider answered (`model_provider`). Set `LlmProvider` to `bedrock` or `anthropic` to use only one. The Anthropic API key is stored as an encrypted SecureString in SSM Parameter Store, and the Lambda role can read only that one parameter.
- **Fail-safe:** if every model provider fails, the ticket is still saved and flagged for a human.

## Ideas for next steps

Authentication on the API (Cognito / JWT), Bedrock Knowledge Bases for larger document sets, a small web UI on S3 + CloudFront, an SQS queue to buffer ticket spikes, and a CloudWatch dashboard for priority mix and latency.
