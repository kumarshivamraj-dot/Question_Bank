from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from study_pipeline.models import Question


ASSET_ROOT = Path("data/processed/original_views")


def _fitz():
    import fitz

    return fitz


def _safe_slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "file"


def _candidate_pdfs(subject: str, json_path: Path) -> list[Path]:
    subject_root = Path("data/uploads") / subject
    if not subject_root.exists():
        return []
    candidates = sorted(subject_root.rglob("*.pdf"))
    if not candidates:
        return []
    same_dir = [path for path in candidates if path.parent == json_path.parent]
    return same_dir + [path for path in candidates if path.parent != json_path.parent]


def find_linked_pdf(subject: str, json_path: Path, pdf_name_hint: str | None = None) -> Path | None:
    candidates = _candidate_pdfs(subject, json_path)
    if not candidates:
        return None
    if pdf_name_hint:
        for path in candidates:
            if path.name.lower() == pdf_name_hint.lower():
                return path
    return candidates[0]


def _asset_path(subject: str, pdf_path: Path, question: Question) -> Path:
    question_key = hashlib.sha1(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "page": question.source_pdf_page,
                "number": question.question_number,
                "text": question.text[:300],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return ASSET_ROOT / _safe_slug(subject) / _safe_slug(pdf_path.stem) / f"{question_key}-page.png"


def _render_page_with_fitz(page, destination: Path, dpi: int = 150) -> str:
    fitz = _fitz()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
    pixmap.save(destination)
    return str(destination)


def _render_page_with_pdftoppm(pdf_path: Path, page_number: int, destination: Path, dpi: int = 150) -> str:
    binary = shutil.which("pdftoppm")
    if not binary:
        raise RuntimeError("pdftoppm is not available")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = destination.with_suffix("")
    subprocess.run(
        [
            binary,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return str(destination)


def _render_pdf_page(pdf_path: Path, page_number: int, destination: Path, dpi: int = 150) -> str | None:
    fitz = None
    document = None
    try:
        fitz = _fitz()
        document = fitz.open(pdf_path)
    except Exception:
        fitz = None
        document = None
    try:
        if document is not None and 1 <= page_number <= len(document):
            return _render_page_with_fitz(document[page_number - 1], destination, dpi=dpi)
        return _render_page_with_pdftoppm(pdf_path, page_number, destination, dpi=dpi)
    except Exception:
        return None
    finally:
        if document is not None:
            document.close()


def materialize_original_asset(
    *,
    subject: str,
    document_path: str,
    question_number: str | None,
    text: str,
    has_diagram: bool,
    source_pdf_name: str | None,
    source_pdf_page: int | None,
) -> tuple[str | None, str | None, str | None]:
    if not source_pdf_page:
        return None, None, source_pdf_name
    json_path = Path(document_path)
    pdf_path = find_linked_pdf(subject, json_path, pdf_name_hint=source_pdf_name)
    if not pdf_path:
        return None, None, source_pdf_name
    question = Question(
        document_path=document_path,
        document_name=json_path.name,
        page_number=0,
        question_number=question_number,
        text=text,
        has_diagram=has_diagram,
        source_pdf_name=pdf_path.name,
        source_pdf_page=int(source_pdf_page),
    )
    destination = _asset_path(subject, pdf_path, question)
    if not destination.exists():
        rendered = _render_pdf_page(pdf_path, int(source_pdf_page), destination)
        if not rendered:
            return None, None, pdf_path.name
    image_path = str(destination)
    return image_path, image_path if has_diagram else None, pdf_path.name


def link_questions_to_pdf(subject: str, json_path: Path, questions: list[Question]) -> list[Question]:
    if not questions:
        return questions

    hint = next((question.source_pdf_name for question in questions if question.source_pdf_name), None)
    pdf_path = find_linked_pdf(subject, json_path, pdf_name_hint=hint)
    if not pdf_path:
        return questions

    for question in questions:
        if not question.source_pdf_page:
            continue
        page_number = int(question.source_pdf_page)
        destination = _asset_path(subject, pdf_path, question)
        image_path = str(destination) if destination.exists() else _render_pdf_page(pdf_path, page_number, destination)
        if not image_path:
            continue
        question.source_pdf_path = str(pdf_path)
        question.source_pdf_name = pdf_path.name
        question.original_image_path = image_path
        question.diagram_image_path = image_path if question.has_diagram else None
        question.link_confidence = 1.0
    return questions
