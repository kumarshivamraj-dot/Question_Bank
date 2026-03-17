from __future__ import annotations

import re
from collections import Counter

from study_pipeline.text_utils import compact_line, looks_like_heading, tokenize


GENERIC_TOPIC_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "define",
    "describe",
    "differentiate",
    "compare",
    "contrast",
    "write",
    "state",
    "list",
    "explain",
    "derive",
    "compute",
    "calculate",
    "draw",
    "discuss",
    "justify",
    "prove",
    "mention",
    "find",
    "give",
    "marks",
    "question",
    "paper",
    "semester",
    "slot",
    "date",
    "winter",
    "summer",
    "wintersem",
    "summersem",
    "wintersemester",
    "summersemester",
}

TOPIC_ALIAS_MAP = {
    "ieee 754": "IEEE-754 Floating Point",
    "booth": "Booth Multiplication Algorithm",
    "booth multiplication": "Booth Multiplication Algorithm",
    "restoring division": "Restoring Division Algorithm",
    "dma": "DMA",
    "cache": "Cache Memory",
    "cache memory": "Cache Memory",
    "direct mapped cache": "Direct-Mapped Cache",
    "set associative cache": "Set-Associative Cache",
    "fully associative cache": "Fully Associative Cache",
    "memory mapped io": "Memory-Mapped I/O",
    "memory mapped i o": "Memory-Mapped I/O",
    "booth multiplication algorithm": "Booth Multiplication Algorithm",
    "floating point": "Floating Point Representation",
    "page faults": "Page Replacement",
    "fifo": "Page Replacement",
    "lru": "Page Replacement",
    "optimal": "Page Replacement",
    "instruction fetch": "Instruction Cycle",
    "control sequence": "Control Sequence",
    "hit miss": "Cache Hit/Miss",
    "hit and miss": "Cache Hit/Miss",
    "tag": "Cache Address Fields",
    "set": "Cache Address Fields",
    "word fields": "Cache Address Fields",
}


def _clean_candidate(text: str) -> str:
    text = compact_line(text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonicalize_topic(topic: str) -> str | None:
    clean = _clean_candidate(topic)
    if not clean:
        return None

    lowered = clean.lower()
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"\b(winter|summer)\s*sem(?:ester)?\b", "", lowered).strip()
    lowered = re.sub(r"\b\d+\s*marks?\b", "", lowered).strip()
    lowered = re.sub(r"\b\d{4}(?:\s+\d{2,4})?\b", "", lowered).strip()
    lowered = re.sub(r"\b(?:paper|semester|slot|date|winter|summer)\b\s*:?", "", lowered).strip()
    lowered = re.sub(r"\s+", " ", lowered)
    if not lowered or lowered in GENERIC_TOPIC_WORDS:
        return None
    if not re.search(r"[a-z]", lowered):
        return None
    if len(lowered) <= 2:
        return None
    if re.fullmatch(r"[a-z]|\d+", lowered):
        return None

    if lowered in TOPIC_ALIAS_MAP:
        return TOPIC_ALIAS_MAP[lowered]

    words = lowered.split()
    if len(words) == 1:
        word = words[0]
        if word in {"mips", "dma", "rom", "ram"}:
            return word.upper()
        if word == "cao":
            return "Computer Architecture and Organization"
        return word.title()

    normalized = " ".join(words)
    if normalized in TOPIC_ALIAS_MAP:
        return TOPIC_ALIAS_MAP[normalized]

    titled = normalized.title()
    titled = titled.replace("Ieee", "IEEE").replace("Dma", "DMA").replace("Io", "I/O")
    titled = titled.replace("Rom", "ROM").replace("Ram", "RAM").replace("Cpu", "CPU")
    titled = titled.replace("Sql", "SQL").replace("Dbms", "DBMS").replace("Os", "OS")
    return titled


def canonicalize_topics(topics: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        canonical = canonicalize_topic(topic)
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(canonical)
    return normalized


def _extract_phrases(text: str) -> list[str]:
    lowered = text.lower()
    phrases: list[str] = []
    patterns = [
        r"\b(?:direct[- ]mapped cache|set[- ]associative cache|fully associative cache)\b",
        r"\b(?:booth'?s? multiplication algorithm|restoring division algorithm)\b",
        r"\b(?:memory[- ]mapped i/?o|instruction fetch phase|control sequence)\b",
        r"\b(?:floating point|ieee[- ]?754|page reference string|page faults?)\b",
        r"\b(?:cache hit(?:/| and )miss|memory address map|chip select|block diagram)\b",
        r"\b(?:page replacement|optimal replacement|single bus architecture)\b",
    ]
    for pattern in patterns:
        phrases.extend(re.findall(pattern, lowered))
    return phrases


def extract_topics(text: str, max_topics: int = 6) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        clean = compact_line(line)
        if looks_like_heading(clean):
            headings.append(clean)

    tokens = tokenize(text)
    common = [word.replace("-", " ") for word, _ in Counter(tokens).most_common(20)]
    phrases = _extract_phrases(text)

    candidates = headings + phrases + common
    topics = canonicalize_topics(candidates)
    return topics[:max_topics]
