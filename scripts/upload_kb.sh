#!/bin/zsh
# Upload the knowledge base articles to the S3 bucket created by the deploy.
# Usage: ./scripts/upload_kb.sh <bucket-name>
set -e
if [ -z "$1" ]; then echo "Usage: ./scripts/upload_kb.sh <bucket-name>"; exit 1; fi
aws s3 sync knowledge_base/ "s3://$1/kb/" --exclude "*" --include "*.md"
echo "Uploaded knowledge base to s3://$1/kb/"
