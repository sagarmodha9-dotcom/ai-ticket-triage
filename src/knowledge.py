"""A small knowledge base of IT help desk articles, and a simple search over them.

This is the "R" in RAG (retrieval-augmented generation): before asking the AI,
we look up the most relevant articles and give them to the model as context.

Search uses TF-IDF keyword scoring. It is cheap, fast, and easy to explain.
(A managed vector database would work too, but costs money every hour it runs.)
"""
import math
import os
import re
from dataclasses import dataclass

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "it", "my", "i",
    "me", "we", "be", "with", "can", "not", "this", "that", "at", "from", "are", "was",
    "but", "have", "has", "do", "does", "how", "what", "when", "you", "your", "our", "if",
    "as", "by", "so", "all", "any", "will", "just", "get", "got", "im", "its", "please",
}


@dataclass
class Article:
    source: str   # file name, e.g. "vpn.md"
    title: str    # first line of the file without the leading "#"
    text: str     # full article text


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _parse(source: str, raw: str) -> Article:
    first_line = raw.strip().splitlines()[0] if raw.strip() else source
    return Article(source=source, title=first_line.lstrip("# ").strip(), text=raw)


def load_from_dir(path: str) -> list[Article]:
    """Load every .md file in a local folder (used for local testing)."""
    articles = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".md"):
            with open(os.path.join(path, name), encoding="utf-8") as f:
                articles.append(_parse(name, f.read()))
    return articles


def load_from_s3(bucket: str, prefix: str, s3_client) -> list[Article]:
    """Load every .md object under a prefix in S3 (used when running in AWS)."""
    articles = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".md"):
                body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
                articles.append(_parse(key.rsplit("/", 1)[-1], body))
    return articles


class KnowledgeBase:
    def __init__(self, articles: list[Article]):
        self.articles = articles
        self._doc_tokens = [tokenize(a.title + " " + a.text) for a in articles]
        n = len(articles)
        doc_freq = {}
        for tokens in self._doc_tokens:
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1
        # Rare words matter more than common ones (inverse document frequency).
        self._idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}

    def search(self, query: str, top_k: int = 3) -> list[Article]:
        q = set(tokenize(query))
        scored = []
        for article, tokens in zip(self.articles, self._doc_tokens):
            if not tokens:
                continue
            score = sum(tokens.count(t) * self._idf.get(t, 0) for t in q) / math.sqrt(len(tokens))
            if score > 0:
                scored.append((score, article))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [a for _, a in scored[:top_k]]
