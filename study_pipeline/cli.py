from __future__ import annotations

import argparse
import json
from pathlib import Path

from study_pipeline.embeddings import HashingEmbeddingProvider, OllamaEmbeddingProvider
from study_pipeline.store import StudyStore


def build_embedding_provider(name: str, ollama_model: str, ollama_url: str):
    if name == "ollama":
        return OllamaEmbeddingProvider(base_url=ollama_url, model=ollama_model)
    return HashingEmbeddingProvider()


def ingest_command(args: argparse.Namespace) -> int:
    from study_pipeline.chunking import chunk_page
    from study_pipeline.extract import extract_document
    from study_pipeline.json_input import load_question_json
    from study_pipeline.questions import extract_questions
    from study_pipeline.vision import extract_pyq_questions_with_vision_fallback

    provider = build_embedding_provider(
        args.embedding_provider,
        args.ollama_model,
        args.ollama_url,
    )
    store = StudyStore(Path(args.db), provider)

    for raw_path in args.paths:
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() == ".json":
            pages, questions = load_question_json(path)
        else:
            pages = extract_document(path, ocr_lang=args.ocr_lang)
            questions = []
            for page in pages:
                if args.source_type == "pyq":
                    questions.extend(
                        extract_questions(
                            page.document_path,
                            page.document_name,
                            page.page_number,
                            page.text,
                        )
                    )
            if args.source_type == "pyq" and path.suffix.lower() == ".pdf":
                questions = extract_pyq_questions_with_vision_fallback(path, questions, args.vision_provider)

        chunks = []
        for page in pages:
            chunks.extend(chunk_page(page, target_size=args.chunk_size))

        store.upsert_document(
            path=str(path),
            name=path.name,
            subject=args.subject,
            kind=path.suffix.lower().lstrip("."),
            source_type=args.source_type,
            chunks=chunks,
            questions=questions,
        )
        print(
            json.dumps(
                {
                    "subject": args.subject,
                    "source_type": args.source_type,
                    "document": path.name,
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "questions": len(questions),
                }
            )
        )
    return 0


def search_command(args: argparse.Namespace) -> int:
    provider = build_embedding_provider(
        args.embedding_provider,
        args.ollama_model,
        args.ollama_url,
    )
    store = StudyStore(Path(args.db), provider)
    results = store.search(args.query, limit=args.limit, subject=args.subject)
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. [{result.subject}] {result.document_name} page {result.page_number} score={result.score:.3f}"
        )
        print(f"   topics: {', '.join(result.topics) if result.topics else '-'}")
        print(f"   path: {result.path}")
        print(f"   text: {result.chunk_text[:400].replace(chr(10), ' ')}")
    return 0


def questions_command(args: argparse.Namespace) -> int:
    provider = build_embedding_provider(
        args.embedding_provider,
        args.ollama_model,
        args.ollama_url,
    )
    store = StudyStore(Path(args.db), provider)
    rows = store.question_search(args.query, limit=args.limit, subject=args.subject)
    for index, row in enumerate(rows, start=1):
        topics = ", ".join(json.loads(row["topics_json"]))
        prefix = ""
        if row["question_number"] and not row["text"].lstrip().lower().startswith(str(row["question_number"]).lower()):
            prefix = f"{row['question_number']} "
        print(
            f"{index}. [{row['subject']}] {row['name']} page {row['page_number']} ({row['source_type']})"
        )
        print(f"   topics: {topics or '-'}")
        print(f"   path: {row['path']}")
        print(f"   text: {prefix}{row['text'][:500].replace(chr(10), ' ')}")
    return 0


def topics_command(args: argparse.Namespace) -> int:
    provider = build_embedding_provider(
        args.embedding_provider,
        args.ollama_model,
        args.ollama_url,
    )
    store = StudyStore(Path(args.db), provider)
    for index, (topic, count) in enumerate(store.subject_topics(args.subject, limit=args.limit), start=1):
        print(f"{index}. {topic} ({count})")
    return 0


def stats_command(args: argparse.Namespace) -> int:
    provider = build_embedding_provider(
        args.embedding_provider,
        args.ollama_model,
        args.ollama_url,
    )
    store = StudyStore(Path(args.db), provider)
    print(json.dumps({"totals": store.stats(), "subjects": [dict(row) for row in store.subject_stats()]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="study-index",
        description="Index class notes, handouts, PPTs, and previous-year papers into one local study database.",
    )
    parser.add_argument("--db", default="data/processed/study_index.db")
    parser.add_argument(
        "--embedding-provider",
        choices=("hashing", "ollama"),
        default="hashing",
    )
    parser.add_argument("--ollama-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index documents into the study database")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--subject", required=True)
    ingest.add_argument(
        "--source-type",
        choices=("pyq", "reference", "handout", "notes", "ppt"),
        required=True,
    )
    ingest.add_argument("--vision-provider", choices=("none", "claude_cheap", "claude"), default="none")
    ingest.add_argument("--ocr-lang", default="eng")
    ingest.add_argument("--chunk-size", type=int, default=1000)
    ingest.set_defaults(func=ingest_command)

    search = subparsers.add_parser("search", help="Vector search across indexed study material")
    search.add_argument("query")
    search.add_argument("--subject")
    search.add_argument("--limit", type=int, default=5)
    search.set_defaults(func=search_command)

    questions = subparsers.add_parser("questions", help="Find questions related to a topic")
    questions.add_argument("query")
    questions.add_argument("--subject")
    questions.add_argument("--limit", type=int, default=10)
    questions.set_defaults(func=questions_command)

    topics = subparsers.add_parser("topics", help="Show important topics for a subject")
    topics.add_argument("--subject", required=True)
    topics.add_argument("--limit", type=int, default=15)
    topics.set_defaults(func=topics_command)

    stats = subparsers.add_parser("stats", help="Show overall and per-subject stats")
    stats.set_defaults(func=stats_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
