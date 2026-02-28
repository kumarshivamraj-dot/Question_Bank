import regex as re
import unicodedata
from typing import List
from collections import Counter


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _is_noise_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return True

    # keep question numbers
    if re.fullmatch(r"(Q?\d+[\.\)]?)", line):
        return False

    # too short to matter
    if len(line) < 4:
        return True

    # alphabetic ratio check
    alpha = sum(c.isalpha() for c in line)
    if alpha / max(len(line), 1) < 0.30:
        return True

    # obvious OCR junk
    if re.search(r"[\\><=%]{3,}", line):
        return True

    return False


def _detect_headers(pages: List[str], threshold: float = 0.35) -> set:
    counter = Counter()
    total = len(pages)

    for page in pages:
        seen = set()
        for line in page.splitlines():
            clean = _normalize(line)
            if clean:
                seen.add(clean)
        for line in seen:
            counter[line] += 1

    return {line for line, count in counter.items() if count / total >= threshold}


def _fix_hyphenation(lines: List[str]) -> List[str]:
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("-") and i + 1 < len(lines):
            nxt = lines[i + 1]
            if nxt and nxt[0].islower():
                fixed.append(line[:-1] + nxt)
                i += 2
                continue
        fixed.append(line)
        i += 1
    return fixed


def post_processing(ocr_text: List[str]) -> List[str]:
    """
    Clean OCR text for:
    - topic extraction
    - embeddings
    - exam-style pattern mining

    Input: list of raw OCR pages
    Output: list of cleaned OCR pages
    """

    headers = _detect_headers(ocr_text)
    cleaned_pages = []

    for page in ocr_text:
        page = _normalize(page)
        lines = page.splitlines()
        cleaned = []

        for line in lines:
            line = _normalize(line)

            if not line:
                continue
            if line in headers:
                continue
            if _is_noise_line(line):
                continue

            # mark question boundaries for downstream AI
            if re.match(r"^(Q?\d+[\.\)])", line):
                cleaned.append("<QUESTION_START>")

            cleaned.append(line)

        cleaned = _fix_hyphenation(cleaned)
        cleaned_pages.append("\n".join(cleaned))

    return cleaned_pages
