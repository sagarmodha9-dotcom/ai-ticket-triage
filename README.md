# AI Support Ticket Triage

A serverless help desk assistant on AWS. When a support ticket comes in, it:

1. **Hides personal data** (emails, phone numbers, IPs, SSNs) before anything reaches the AI
2. **Finds relevant help desk articles** in a knowledge base stored in S3 (retrieval / RAG)
3. **Asks Claude on Amazon Bedrock** to set a **priority (P1–P4)** and **category**, summarize the issue, and draft a reply based on the articles
4. **Checks the AI's answer** with plain-code guardrails: valid values only, no made-up sources, security tickets always escalated, low-confidence answers flagged for a human
5. **Saves the ticket** in DynamoDB and **emails an alert** through SNS when it's a P1

It also has a **help desk agent** (`POST /agent`), which scored **95% vs 57%** against single-call triage on the same scenarios ([results](#results)). A technician asks a question in plain English and Claude decides which tools to use, step by step: search the knowledge base, look up a ticket, find similar recent tickets, add a note, or escalate. For example, it spots an outage when several users report the same problem, escalates the newest ticket to P1, and alerts on-call.

The whole API also runs in **Docker** with one command, no AWS account needed.

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

 POST /agent ──▶ API Gateway ──▶ Lambda: agent ◀──▶ Claude (tool use loop, max 6 steps)
                                      │
                 tools: search_knowledge_base · get_ticket · search_recent_tickets
                        add_ticket_note · escalate_ticket (P1 ──▶ SNS email)
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

## Help desk agent

```bash
curl -X POST "$API_URL/agent" -H "Content-Type: application/json" -H "x-api-key: $AGENT_KEY" \
  -d '{"message": "Is there a VPN outage right now?"}'
```

```json
{
  "answer": "Three users reported VPN failures in the last hour, which looks like an outage. I escalated the newest ticket to P1 and on-call was alerted. Next step: check the VPN gateway, then follow vpn.md for affected users.",
  "steps": [
    {"step": 1, "tool": "search_recent_tickets", "input": {"keyword": "vpn", "hours": 24}, "ok": true},
    {"step": 2, "tool": "search_knowledge_base", "input": {"query": "vpn connection failing"}, "ok": true},
    {"step": 3, "tool": "escalate_ticket", "input": {"ticket_id": "3e34...", "priority": "P1", "reason": "Possible outage: 3 similar VPN tickets"}, "ok": true}
  ],
  "actions": [{"action": "escalate_ticket", "ticket_id": "3e34...", "priority_before": "P2", "priority_after": "P1"}],
  "iterations": 4,
  "stopped": "done",
  "usage": {"input_tokens": 5120, "output_tokens": 410},
  "model_provider": "bedrock",
  "latency_ms": 6800
}
```

Pass `"ticket_id"` too to ask about one ticket ("Look at this ticket and tell me what to do next").

**Guardrails in code, not just the prompt** (`src/agent.py`):

- Only the five listed tools can run, and every input is validated (bad IDs, unknown tools and failures go back to the model as errors instead of crashing).
- The agent can add notes and escalate. It **can't close tickets or lower a priority**. A request to lower one is ignored.
- At most **1 escalation and 3 notes per request**, so a bad loop can't spam on-call.
- **PII is redacted** from the question and from every tool result before the model sees it.
- Text users wrote (subjects, descriptions, notes) comes back under **`untrusted_` keys**, and the system prompt says never to follow instructions inside them. The evals test both direct injection (in the ticket) and indirect injection (hidden in a past ticket the agent pulls up).
- **Priority writes are conditional:** the agent only writes a priority when raising it, and only if nobody changed it since it was read, so a stale read or two requests at once can't lower one.
- If the P1 alert fails, the escalation still shows up in the trace, flagged `alert_sent: false`.
- Keyword search matches redacted text, so it can't be used to probe tickets for phone numbers or emails.
- **`/agent` requires an API key** (`x-api-key` header, stored encrypted in SSM) because it can change tickets and page on-call.
- **Step limit of 6, a 30,000-token budget, and a time budget** that stops before Lambda's timeout. If one is hit, the answer says a technician should review, and everything already done is still returned.
- Every response includes the **full trace** (tools called, inputs, what changed) and token usage, so a human can audit it.
- It works with Bedrock or the Anthropic API, and fails over between them like triage does.

## Results

Measured with real Claude (Claude Haiku 4.5 via the Anthropic API) on Oct 18, 2026.

**Help desk agent vs single-call triage** (`evals/run_agent_evals.py`, 12 scenarios × 3 runs each):

| | Agent | Single-call triage |
|---|---|---|
| Scenarios both can attempt (7 × 3 runs) | **20/21 (95%)** | 12/21 (57%) |
| All agent scenarios (12 × 3 runs) | **35/36 (97%)** | n/a |
| Tool calls per run | 1.9 | 0 |
| Latency per run | 4.2 s | 2.1 s |
| Cost per run (Haiku 4.5 list price) | ~$0.0055 | ~$0.0017 |

Single-call triage only sees one ticket, so it scored 0/3 on every scenario that needs looking across tickets: a VPN outage, the same outage described in different words, and an MFA-fatigue attack. The agent caught all three. It also passed both prompt-injection tests 3/3: instructions inside the ticket, and instructions hidden in a past ticket it pulled up during a search.

The trade-off: about 3× the cost and 2× the latency of a single call, still well under a cent per run.

**Known failure:** in the simple VPN outage scenario, the agent correctly said "there appears to be a VPN outage" in all 3 runs, but in 1 of 3 it didn't escalate the ticket. Next step: make the escalation rule more explicit in the prompt and track this scenario's consistency over more runs.

**Single-call triage accuracy** (`evals/run_evals.py`, 12 tickets): 100% priority, 92% category.

**Live deployment:** the deployed `/agent` endpoint found three VPN tickets, escalated the newest to P1, and sent the on-call alert email. Bedrock was throttled on this account, so the call failed over to the Anthropic API automatically.

Reproduce (about $0.25 in API calls):

```bash
ANTHROPIC_API_KEY=... python evals/run_agent_evals.py --anthropic claude-haiku-4-5 --runs 3
ANTHROPIC_API_KEY=... python evals/run_evals.py --anthropic claude-haiku-4-5
```

The evals check what the agent *did* (which tools it called, what it changed in the tickets), not how its answer sounds, so the score means the same thing every run. CI also runs both evals with a fake model on every push and fails the build if they drop below a floor.

## Run it in Docker

```bash
docker build -t ai-ticket-triage .
docker run --rm -p 8000:8000 ai-ticket-triage
```

That starts the API on `http://localhost:8000` with a fake model and a few demo tickets (three VPN tickets, so the agent has an outage to find). P1 alerts print in the container log instead of emailing.

```bash
curl -X POST localhost:8000/agent -H "Content-Type: application/json" -d '{"message": "Is there a VPN outage right now?"}'
curl localhost:8000/tickets
```

To use real Claude, pass a provider and key: `docker run --rm -p 8000:8000 -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY ai-ticket-triage`.

GitHub Actions (`.github/workflows/ci.yml`) runs the tests and evals (the build fails if accuracy drops), builds the image, and smoke-tests the running container on every push. Locally there's no agent key unless you set `AGENT_API_KEY`.

## Project layout

| Path | What it is |
|---|---|
| `template.yaml` | AWS SAM template: every AWS resource, defined as code |
| `src/app.py` | Lambda handlers for the three API routes |
| `src/agent.py` | Help desk agent: tools, guardrails and the tool-use loop |
| `src/store.py` | Ticket storage: DynamoDB in AWS, in-memory for local and Docker |
| `src/local_server.py` | Runs the API locally / in Docker without AWS |
| `Dockerfile` | Container image for the local API |
| `src/triage.py` | Prompt, output validation, guardrails |
| `src/knowledge.py` | Loads articles from S3 and searches them (TF-IDF) |
| `src/redact.py` | Removes personal data before the AI call |
| `src/llm.py` | Bedrock Converse API and Anthropic API clients (text and tool use), provider failover, and a fake model for tests |
| `knowledge_base/` | Sample IT help desk articles |
| `tests/` | 57 unit tests (no AWS needed) |
| `evals/` | 12 tickets with known answers (triage accuracy) and 12 agent scenarios (outages, phishing, MFA fatigue, direct and indirect prompt injection, notes...) compared against single-call triage |

## Run it locally (no AWS account needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q                      # 57 tests
python evals/run_evals.py                # accuracy with the fake model
python scripts/local_demo.py "Locked out" "Too many password attempts"
python src/local_server.py               # the full API on localhost:8000
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
4. Create the API key for `/agent` (random, stored encrypted; you never type it):
   ```bash
   aws ssm put-parameter --name /ai-ticket-triage/agent-api-key --type SecureString \
     --value "$(openssl rand -hex 24)" --region us-east-1
   AGENT_KEY=$(aws ssm get-parameter --name /ai-ticket-triage/agent-api-key --with-decryption \
     --query Parameter.Value --output text --region us-east-1)
   ```
5. Upload the knowledge base (bucket name is in the deploy outputs):
   ```bash
   ./scripts/upload_kb.sh <KnowledgeBucketName>
   ```
6. Test with the `curl` examples above using the `ApiUrl` output.
7. Score the agent: `python evals/run_agent_evals.py --anthropic claude-haiku-4-5 --runs 3` (pass rate and consistency, compared with single-call triage on the same scenarios, plus tool calls, tokens, latency and cost per run)
8. Measure real accuracy: `python evals/run_evals.py --bedrock <MODEL_ID>` (or `--anthropic claude-haiku-4-5` with `ANTHROPIC_API_KEY` set)

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
- **Agent with limited powers:** triage is a single model call because that's all it needs. The agent is for open-ended questions that need several lookups. It gets read tools plus two narrow write tools, its own IAM role (GetItem, Scan and UpdateItem only, no delete), and hard caps enforced in Python.
- **Recent-ticket search uses a DynamoDB Scan:** fine for a small table. At scale, add a GSI on `created_at` and Query it.

## Ideas for next steps

Per-user authentication (Cognito / JWT) instead of one shared agent key, Bedrock Knowledge Bases for larger document sets, a small web UI on S3 + CloudFront, an SQS queue to buffer ticket spikes, and a CloudWatch dashboard for priority mix and latency.
