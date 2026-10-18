# Runs the whole API (ticket triage + help desk agent) in one container, no AWS account needed.
#
#   docker build -t ai-ticket-triage .
#   docker run --rm -p 8000:8000 ai-ticket-triage                 # fake model, free
#   docker run --rm -p 8000:8000 -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY ai-ticket-triage

FROM python:3.12-slim

# Logs show up right away; no .pyc files in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# boto3 is only needed for the Bedrock provider (Lambda already includes it).
# Installed in its own layer so code changes don't re-download it.
RUN pip install "boto3>=1.35"

COPY src/ ./src/
COPY knowledge_base/ ./knowledge_base/

# Don't run as root inside the container.
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV PORT=8000 \
    KB_LOCAL_DIR=/app/knowledge_base \
    LLM_PROVIDER=fake \
    SEED_DEMO_TICKETS=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["python", "src/local_server.py"]
