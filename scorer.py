import re


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match_score(predicted: str, gold: str) -> bool:
    return normalize_text(predicted) == normalize_text(gold)
