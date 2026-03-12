from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PageContent:
    document_path: str
    document_name: str
    page_number: int
    text: str
    extraction_method: str


@dataclass(slots=True)
class Chunk:
    document_path: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    topics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Question:
    document_path: str
    document_name: str
    page_number: int
    question_number: str | None
    text: str
    topics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    score: float
    subject: str
    document_name: str
    page_number: int
    chunk_text: str
    topics: list[str]
    path: str
