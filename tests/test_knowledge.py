from conftest import KB_DIR
from knowledge import KnowledgeBase, load_from_dir, tokenize


def kb():
    return KnowledgeBase(load_from_dir(KB_DIR))


def test_tokenize_drops_stopwords():
    assert tokenize("My VPN is not working") == ["vpn", "working"]


def test_vpn_question_finds_vpn_article():
    results = kb().search("VPN keeps disconnecting at the hotel")
    assert results[0].source == "vpn.md"


def test_phishing_question_finds_security_article():
    results = kb().search("I clicked a link in a suspicious email")
    assert results[0].source == "phishing-security.md"


def test_unrelated_question_returns_nothing():
    assert kb().search("zzzz qqqq") == []
