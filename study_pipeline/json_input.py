from __future__ import annotations

import json
from pathlib import Path

from study_pipeline.models import PageContent, Question
from study_pipeline.topics import extract_topics
from study_pipeline.text_utils import normalize_text


def _normalize_subparts(subparts: list[dict]) -> str:
    lines = []
    for item in subparts:
        label = str(item.get("label") or "").strip()
        text = normalize_text(str(item.get("text") or ""))
        if not text:
            continue
        prefix = f"({label}) " if label else ""
        lines.append(f"{prefix}{text}".strip())
    return "\n".join(lines)


def _question_text(item: dict) -> str:
    main_text = normalize_text(
        str(item.get("text") or item.get("question") or "")
    )
    subparts_text = _normalize_subparts(item.get("subparts") or [])
    if main_text and subparts_text:
        return f"{main_text}\n{subparts_text}"
    return main_text or subparts_text


def _iter_questions(payload: object) -> list[tuple[dict, dict | None]]:
    if isinstance(payload, list):
        return [(item, None) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("questions"), list):
        return [
            (item, None)
            for item in payload["questions"]
            if isinstance(item, dict)
        ]

    rows: list[tuple[dict, dict | None]] = []
    for paper in payload.get("papers", []):
        if not isinstance(paper, dict):
            continue
        for item in paper.get("questions", []):
            if isinstance(item, dict):
                rows.append((item, paper))
    return rows


def load_question_json(path: Path) -> tuple[list[PageContent], list[Question]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pages: list[PageContent] = []
    questions: list[Question] = []
    for index, (item, paper) in enumerate(_iter_questions(payload), start=1):
        text = _question_text(item)
        if not text:
            continue

        topics = item.get("topics")
        if not isinstance(topics, list) or not topics:
            topics = extract_topics(text, max_topics=4)
        topics = [str(topic) for topic in topics]

        question_number = (
            item.get("question_number")
            or item.get("number")
            or item.get("id")
        )
        page_number = int(item.get("page") or index)
        page_text = text

        pages.append(
            PageContent(
                document_path=str(path),
                document_name=path.name,
                page_number=page_number,
                text=page_text,
                extraction_method="json_import",
            )
        )
        questions.append(
            Question(
                document_path=str(path),
                document_name=path.name,
                page_number=page_number,
                question_number=str(question_number) if question_number is not None else None,
                text=text,
                topics=topics,
            )
        )
    return pages, questions
