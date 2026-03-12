from __future__ import annotations

import re

from study_pipeline.models import Question
from study_pipeline.topics import extract_topics
from study_pipeline.text_utils import compact_line, normalize_text


TOP_LEVEL_QUESTION_PATTERN = re.compile(
    r"(?i)^(?:question\s*|q\s*)?(?P<number>\d{1,3})(?:[.):-])\s+"
)
SUBPART_PATTERN = re.compile(
    r"(?i)^(?P<label>\(?[a-z]\)|\(?[ivxlcdm]+\)|\(?\d+\))\s+"
)
QUESTION_INTENT_PATTERN = re.compile(
    r"(?i)\b(?:what|why|how|when|where|which|who|define|describe|differentiate|compare|contrast|write|state|list|explain|derive|compute|calculate|draw|discuss|justify|prove|mention|find|give)\b"
)
NOISE_LINE_PATTERN = re.compile(
    r"(?i)\b(?:solution|maximum marks|max\.?\s*marks|slot|school of|semester|institute of technology|continuous assessment|part [a-d]|answer all questions)\b"
)
SOLUTION_START_PATTERN = re.compile(
    r"(?i)^(?:solution|soln\.?|answer|ans\.?|method|approach|steps?|procedure|working|proof|explanation|given|to solve|algorithm)\b"
)
SOLUTION_SIGNAL_PATTERN = re.compile(
    r"(?i)\b(?:solution|soln\.?|answer|ans\.?|therefore|hence|thus|substituting|let us|we have|formula|using|step\s*\d+|calculation|working|proof)\b"
)


def _looks_like_noise(line: str) -> bool:
    if not line:
        return True
    if NOISE_LINE_PATTERN.search(line):
        return True
    non_space = [char for char in line if not char.isspace()]
    if not non_space:
        return True
    alpha_count = sum(char.isalpha() for char in non_space)
    if alpha_count / len(non_space) < 0.45:
        return True
    words = re.findall(r"[A-Za-z]{2,}", line)
    if len(words) < 3:
        return True
    short_words = sum(len(word) <= 2 for word in words)
    return short_words / len(words) > 0.55


def _candidate_quality(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 35 or len(stripped) > 900:
        return False
    letters = sum(char.isalpha() for char in stripped)
    non_space = sum(not char.isspace() for char in stripped)
    if not non_space or letters / non_space < 0.6:
        return False
    words = re.findall(r"[A-Za-z]{2,}", stripped)
    if len(words) < 6:
        return False
    if not QUESTION_INTENT_PATTERN.search(stripped) and "?" not in stripped:
        return False
    if NOISE_LINE_PATTERN.search(stripped):
        return False
    if looks_like_solution_block(stripped):
        return False
    return True


def looks_like_solution_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if SOLUTION_START_PATTERN.search(stripped):
        return True
    signal_hits = len(SOLUTION_SIGNAL_PATTERN.findall(stripped))
    intent_hits = len(QUESTION_INTENT_PATTERN.findall(stripped))
    if signal_hits >= 2 and intent_hits == 0 and "?" not in stripped:
        return True
    return False


def _collect_question_blocks(text: str) -> list[tuple[str | None, str]]:
    lines = [compact_line(line) for line in normalize_text(text).splitlines()]
    blocks: list[tuple[str | None, list[str]]] = []
    current_number: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if not line or _looks_like_noise(line):
            continue

        if SOLUTION_START_PATTERN.match(line):
            if current_lines:
                blocks.append((current_number, current_lines))
            current_number = None
            current_lines = []
            continue

        top_level_match = TOP_LEVEL_QUESTION_PATTERN.match(line)
        if top_level_match:
            if current_lines:
                blocks.append((current_number, current_lines))
            current_number = top_level_match.group("number")
            current_lines = [line]
            continue

        subpart_match = SUBPART_PATTERN.match(line)
        if subpart_match and current_lines:
            current_lines.append(line)
            continue

        if current_lines:
            if looks_like_solution_block(line):
                blocks.append((current_number, current_lines))
                current_number = None
                current_lines = []
                continue
            current_lines.append(line)

    if current_lines:
        blocks.append((current_number, current_lines))

    return [(number, " ".join(lines)) for number, lines in blocks]


def extract_questions(document_path: str, document_name: str, page_number: int, text: str) -> list[Question]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    questions: list[Question] = []
    for question_number, candidate in _collect_question_blocks(normalized):
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not _candidate_quality(candidate):
            continue
        questions.append(
            Question(
                document_path=document_path,
                document_name=document_name,
                page_number=page_number,
                question_number=question_number,
                text=candidate,
                topics=extract_topics(candidate, max_topics=4),
            )
        )
    return questions
