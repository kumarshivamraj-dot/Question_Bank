from __future__ import annotations

import re
import unicodedata


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}

ACADEMIC_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"(?i)^\s*vit\b.*\bvellore\b.*$"),
    re.compile(r"(?i)^\s*vellore institute of technology\b.*$"),
    re.compile(r"(?i)^\s*(winter|summer|fall|spring)\s+\d{4}(?:\s*-\s*\d{2,4})?.*$"),
    re.compile(r"(?i)^\s*slot\s*[:\-]?\s*[a-z0-9+]+.*$"),
    re.compile(r"(?i)^\s*[a-z]\d(?:\+t[a-z]?\d)?\s*$"),
    re.compile(r"(?i)^\s*[a-z]{2,}\d{3,}[a-z]?\s*$"),
    re.compile(r"(?i)^\s*(paper|course code|course title|semester|date)\s*[:\-].*$"),
]

ACADEMIC_BOILERPLATE_INLINE_PATTERNS = [
    re.compile(r"(?i)\bvit\b\s*,?\s*\bvellore\b"),
    re.compile(r"(?i)\bvellore institute of technology\b"),
    re.compile(r"(?i)\b(?:winter|summer|fall|spring)\s+\d{4}(?:\s*-\s*\d{2,4})?\b"),
    re.compile(r"(?i)\bslot\s*[:\-]?\s*[a-z0-9+]+\b"),
    re.compile(r"(?i)\b[a-z]\d(?:\+t[a-z]?\d)?\b"),
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x0c", "\n")
    text = strip_academic_boilerplate(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_academic_boilerplate(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = unicodedata.normalize("NFKC", raw_line or "").strip()
        if not line:
            cleaned_lines.append("")
            continue
        if any(pattern.match(line) for pattern in ACADEMIC_BOILERPLATE_LINE_PATTERNS):
            continue
        normalized = line
        for pattern in ACADEMIC_BOILERPLATE_INLINE_PATTERNS:
            normalized = pattern.sub(" ", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip(" ,:-")
        if normalized:
            cleaned_lines.append(normalized)
    return "\n".join(cleaned_lines)


def compact_line(line: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(line))


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", text.lower())
        if token not in STOPWORDS
    ]


def looks_like_heading(line: str) -> bool:
    clean = compact_line(line)
    if not clean or len(clean) > 90:
        return False
    letters = [char for char in clean if char.isalpha()]
    if len(letters) < 4:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    title_case = sum(word[:1].isupper() for word in clean.split()) >= max(
        1, len(clean.split()) - 1
    )
    return uppercase_ratio > 0.7 or title_case
