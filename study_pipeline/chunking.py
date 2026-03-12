from __future__ import annotations

from study_pipeline.models import Chunk, PageContent
from study_pipeline.topics import extract_topics
from study_pipeline.text_utils import normalize_text


def chunk_page(page: PageContent, target_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    text = normalize_text(page.text)
    if not text:
        return []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[Chunk] = []
    current = ""
    chunk_index = 0

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > target_size:
            chunks.append(
                Chunk(
                    document_path=page.document_path,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=current,
                    topics=extract_topics(current),
                )
            )
            chunk_index += 1
            current = current[-overlap:].strip()
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
        else:
            current = candidate

    if current:
        chunks.append(
            Chunk(
                document_path=page.document_path,
                document_name=page.document_name,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=current,
                topics=extract_topics(current),
            )
        )

    return chunks

