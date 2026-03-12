from __future__ import annotations

import tempfile
from pathlib import Path

from study_pipeline.models import Question
from study_pipeline.questions import looks_like_solution_block
from study_pipeline.topics import extract_topics


def extract_pyq_questions_with_vision(path: Path, provider: str) -> list[Question]:
    if provider == "none":
        return []
    if provider not in {"claude", "claude_cheap"}:
        raise ValueError(f"Unsupported vision provider: {provider}")

    import fitz

    from ai_processor.claude_vision import claude_extract_questions, claude_extract_questions_cheap

    document = fitz.open(path)
    questions: list[Question] = []
    with tempfile.TemporaryDirectory(prefix="pyq-pages-") as temp_dir:
        temp_root = Path(temp_dir)
        for page_number, page in enumerate(document, start=1):
            image_path = temp_root / f"page_{page_number:03d}.png"
            pixmap = page.get_pixmap(dpi=220, alpha=False)
            pixmap.save(image_path)
            extractor = claude_extract_questions_cheap if provider == "claude_cheap" else claude_extract_questions
            extracted = extractor(str(image_path), path.name, page_number)
            for item in extracted.get("questions", []):
                text = (item.get("text") or "").strip()
                if not text or looks_like_solution_block(text):
                    continue
                questions.append(
                    Question(
                        document_path=str(path),
                        document_name=path.name,
                        page_number=page_number,
                        question_number=str(item.get("number")) if item.get("number") is not None else None,
                        text=text,
                        topics=extract_topics(text, max_topics=4),
                    )
                )
    return questions


def extract_pyq_questions_with_vision_fallback(
    path: Path,
    existing_questions: list[Question],
    provider: str,
) -> list[Question]:
    if provider == "none" or existing_questions:
        return existing_questions
    return extract_pyq_questions_with_vision(path, provider)
