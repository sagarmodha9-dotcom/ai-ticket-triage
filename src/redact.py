"""Remove personal data from ticket text before it is sent to the AI model.

Support tickets often contain emails, phone numbers, and IP addresses.
We replace them with placeholders so private data never leaves our system.
"""
import re

# Each pattern is (placeholder, compiled regex). Order matters: more specific first.
_PATTERNS = [
    ("<SSN>", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("<EMAIL>", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("<IP>", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("<PHONE>", re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")),
]


def redact(text: str) -> tuple[str, dict]:
    """Return (redacted_text, counts) where counts says how many of each type were hidden."""
    counts = {}
    for placeholder, pattern in _PATTERNS:
        text, n = pattern.subn(placeholder, text)
        if n:
            counts[placeholder.strip("<>").lower()] = n
    return text, counts
