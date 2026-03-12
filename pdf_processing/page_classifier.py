"""
page_classifier.py
Heuristic classification of pages based on messy OCR text.
Returns: "question" | "no_question" | "maybe_question"
"""

import re


QUESTION_PATTERNS = [
    r"\bQ\s*\d+[\.\)]",      # Q1. or Q 1)
    r"^\s*\d+[\.\)]",        # 1. or 1) at line start
    r"^\s*\d+\.\s+",         # 1. at start
    r"\b(section\s+[A-Z])\b",
    r"\?",                   # literal question mark
    r"\*\s*\?\s*\*",         # *?* (OCR corruption of question mark)
]


def classify_page_heuristic(text: str) -> str:
    """
    Classify a page based on messy OCR text.

    Returns one of:
      - "question": strong evidence of exam questions (do not need Claude keep/drop)
      - "no_question": strong evidence of junk/empty (skip Claude entirely)
      - "maybe_question": ambiguous (send to Claude keep/drop for final decision)
    """
    t = text or ""
    t = t.strip()

    if not t:
        return "no_question"

    length = len(t)
    alpha = sum(c.isalpha() for c in t)
    alpha_ratio = alpha / max(length, 1)

    has_qmark = "?" in t

    pattern_hits = sum(
        1
        for pat in QUESTION_PATTERNS
        if re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
    )

    # Very short or almost no letters -> junk (unless question markers)
    if length < 80 or alpha_ratio < 0.25:
        if not has_qmark and pattern_hits == 0:
            return "no_question"

    # Strong evidence of questions
    if has_qmark or pattern_hits >= 2:
        if alpha_ratio >= 0.35 and length >= 120:
            return "question"

    # Borderline: could be noisy OCR of questions or junk
    return "maybe_question"


def make_page_id(pdf: str, page: int) -> str:
    """Canonical page identifier, consistent across all pipeline stages."""
    return f"{pdf}::page_{page:03d}"


def make_image_path(processed_dir: str, pdf: str, page: int) -> str:
    """Resolve image path from pdf name and page number (1-indexed)."""
    return f"{processed_dir}/{pdf}/page_{page:03d}.png"
