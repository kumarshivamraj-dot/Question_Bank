from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from study_pipeline.embeddings import EmbeddingProvider, cosine_similarity
from study_pipeline.models import Chunk, Question, SearchResult
from study_pipeline.topics import canonicalize_topic, canonicalize_topics


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT 'general',
    kind TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'notes',
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    topics_json TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    question_number TEXT,
    text TEXT NOT NULL,
    topics_json TEXT NOT NULL,
    embedding_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
    text,
    content='questions',
    content_rowid='id'
);
"""


class StudyStore:
    def __init__(self, db_path: Path, embedding_provider: EmbeddingProvider) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self._migrate_existing_schema()

    def _migrate_existing_schema(self) -> None:
        self._ensure_column("documents", "subject", "TEXT NOT NULL DEFAULT 'general'")
        self._ensure_column("documents", "source_type", "TEXT NOT NULL DEFAULT 'notes'")
        self._ensure_column("questions", "embedding_json", "TEXT NOT NULL DEFAULT '[]'")
        self.connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
                text,
                content='questions',
                content_rowid='id'
            )
            """
        )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def upsert_document(
        self,
        path: str,
        name: str,
        subject: str,
        kind: str,
        source_type: str,
        chunks: list[Chunk],
        questions: list[Question],
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO documents(path, name, subject, kind, source_type) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "name=excluded.name, subject=excluded.subject, kind=excluded.kind, source_type=excluded.source_type",
                (path, name, subject, kind, source_type),
            )
            document_id = self.connection.execute(
                "SELECT id FROM documents WHERE path = ?",
                (path,),
            ).fetchone()["id"]

            question_ids = self.connection.execute(
                "SELECT id FROM questions WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            for row in question_ids:
                self.connection.execute(
                    "DELETE FROM questions_fts WHERE rowid = ?",
                    (row["id"],),
                )
            self.connection.execute("DELETE FROM questions WHERE document_id = ?", (document_id,))

            chunk_ids = self.connection.execute(
                "SELECT id FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            for row in chunk_ids:
                self.connection.execute(
                    "DELETE FROM chunks_fts WHERE rowid = ?",
                    (row["id"],),
                )
            self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

            for chunk in chunks:
                normalized_topics = canonicalize_topics(chunk.topics)
                embedding = self.embedding_provider.embed(chunk.text)
                cursor = self.connection.execute(
                    "INSERT INTO chunks(document_id, page_number, chunk_index, text, topics_json, embedding_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        document_id,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(normalized_topics),
                        json.dumps(embedding),
                    ),
                )
                self.connection.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, chunk.text),
                )

            for question in questions:
                normalized_topics = canonicalize_topics(question.topics)
                embedding = self.embedding_provider.embed(question.text)
                cursor = self.connection.execute(
                    "INSERT INTO questions(document_id, page_number, question_number, text, topics_json, embedding_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        document_id,
                        question.page_number,
                        question.question_number,
                        question.text,
                        json.dumps(normalized_topics),
                        json.dumps(embedding),
                    ),
                )
                self.connection.execute(
                    "INSERT INTO questions_fts(rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, question.text),
                )

    def search(
        self,
        query: str,
        limit: int = 5,
        subject: str | None = None,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_provider.embed(query)
        fts_scores = {
            row["rowid"]: row["keyword_score"]
            for row in self.connection.execute(
                """
                SELECT rowid, bm25(chunks_fts) AS keyword_score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                """,
                (query,),
            ).fetchall()
        }
        rows = self.connection.execute(
            """
            SELECT
                chunks.id,
                documents.name,
                documents.subject,
                documents.path,
                chunks.page_number,
                chunks.text,
                chunks.topics_json,
                chunks.embedding_json
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE (? IS NULL OR documents.subject = ?)
            """,
            (subject, subject),
        ).fetchall()

        scored: list[SearchResult] = []
        for row in rows:
            vector_score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            keyword_score = float(fts_scores.get(row["id"], 0.0))
            final_score = vector_score + (0.15 / (1.0 + abs(keyword_score)))
            scored.append(
                SearchResult(
                    score=final_score,
                    subject=row["subject"],
                    document_name=row["name"],
                    page_number=row["page_number"],
                    chunk_text=row["text"],
                    topics=json.loads(row["topics_json"]),
                    path=row["path"],
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def question_search(
        self,
        query: str,
        limit: int = 10,
        subject: str | None = None,
    ) -> list[sqlite3.Row]:
        query_embedding = self.embedding_provider.embed(query)
        rows = self.connection.execute(
            """
            SELECT documents.name, documents.path, documents.subject, documents.source_type,
                   questions.page_number, questions.question_number, questions.text,
                   questions.topics_json, questions.embedding_json
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE (? IS NULL OR documents.subject = ?)
            """,
            (subject, subject),
        ).fetchall()

        scored = []
        query_lower = query.lower()
        for row in rows:
            vector_score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            topics = [topic.lower() for topic in json.loads(row["topics_json"])]
            keyword_bonus = 0.2 if query_lower in row["text"].lower() else 0.0
            if any(query_lower in topic for topic in topics):
                keyword_bonus += 0.3
            scored.append((vector_score + keyword_bonus, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def subject_topics(self, subject: str, limit: int = 15) -> list[tuple[str, int]]:
        rows = self.connection.execute(
            """
            SELECT chunks.topics_json, chunks.text
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.subject = ?
            """,
            (subject,),
        ).fetchall()
        counts: dict[str, float] = {}
        for row in rows:
            seen_in_chunk: set[str] = set()
            for topic in canonicalize_topics(json.loads(row["topics_json"])):
                key = topic.lower()
                if key in seen_in_chunk:
                    continue
                seen_in_chunk.add(key)
                bonus = 0.15 if topic.lower() in row["text"].lower() else 0.0
                counts[topic] = counts.get(topic, 0.0) + 1.0 + bonus
        return [
            (topic, round(score, 2))
            for topic, score in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    def question_topic_counts(self, subject: str, limit: int = 20) -> list[tuple[str, int]]:
        rows = self.connection.execute(
            """
            SELECT questions.topics_json, questions.text
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
            """,
            (subject,),
        ).fetchall()
        counts: dict[str, float] = {}
        for row in rows:
            seen_in_question: set[str] = set()
            for topic in canonicalize_topics(json.loads(row["topics_json"])):
                key = topic.lower()
                if key in seen_in_question:
                    continue
                seen_in_question.add(key)
                explicit_bonus = 0.35 if topic.lower() in row["text"].lower() else 0.0
                counts[topic] = counts.get(topic, 0.0) + 2.0 + explicit_bonus
        return [
            (topic, round(score, 2))
            for topic, score in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    def important_topics(self, subject: str, limit: int = 20) -> list[dict[str, object]]:
        question_rows = self.connection.execute(
            """
            SELECT questions.topics_json, questions.text
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
            """,
            (subject,),
        ).fetchall()
        chunk_rows = self.connection.execute(
            """
            SELECT chunks.topics_json, chunks.text
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.subject = ?
            """,
            (subject,),
        ).fetchall()

        aggregate: dict[str, dict[str, object]] = {}

        def ensure(topic: str) -> dict[str, object]:
            return aggregate.setdefault(
                topic.lower(),
                {
                    "topic": topic,
                    "score": 0.0,
                    "question_count": 0,
                    "chunk_count": 0,
                    "examples": [],
                },
            )

        for row in question_rows:
            seen: set[str] = set()
            for topic in canonicalize_topics(json.loads(row["topics_json"])):
                key = topic.lower()
                if key in seen:
                    continue
                seen.add(key)
                item = ensure(topic)
                item["score"] += 2.0 + (0.35 if key in row["text"].lower() else 0.0)
                item["question_count"] += 1
                if len(item["examples"]) < 2:
                    item["examples"].append(row["text"][:180].strip())

        for row in chunk_rows:
            seen: set[str] = set()
            for topic in canonicalize_topics(json.loads(row["topics_json"])):
                key = topic.lower()
                if key in seen:
                    continue
                seen.add(key)
                item = ensure(topic)
                item["score"] += 1.0 + (0.15 if key in row["text"].lower() else 0.0)
                item["chunk_count"] += 1

        ranked = sorted(
            aggregate.values(),
            key=lambda item: (
                item["question_count"],
                item["chunk_count"],
                item["score"],
                str(item["topic"]).lower(),
            ),
            reverse=True,
        )[:limit]
        return [
            {
                "topic": item["topic"],
                "score": round(float(item["score"]), 2),
                "question_count": int(item["question_count"]),
                "chunk_count": int(item["chunk_count"]),
                "examples": item["examples"],
            }
            for item in ranked
            if canonicalize_topic(str(item["topic"])) and int(item["question_count"]) > 0
        ]

    def subject_questions(self, subject: str, limit: int = 100) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT documents.name, documents.path, documents.subject, documents.source_type,
                   questions.page_number, questions.question_number, questions.text,
                   questions.topics_json
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
            ORDER BY documents.name, questions.page_number, questions.id
            LIMIT ?
            """,
            (subject, limit),
        ).fetchall()

    def questions_for_topic(
        self,
        subject: str,
        topic: str,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        canonical_topic = canonicalize_topic(topic)
        if not canonical_topic:
            return []

        rows = self.connection.execute(
            """
            SELECT documents.name, documents.path, documents.subject, documents.source_type,
                   questions.page_number, questions.question_number, questions.text,
                   questions.topics_json
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
            ORDER BY documents.name, questions.page_number, questions.id
            """,
            (subject,),
        ).fetchall()

        matches: list[sqlite3.Row] = []
        canonical_key = canonical_topic.lower()
        for row in rows:
            topics = canonicalize_topics(json.loads(row["topics_json"]))
            if any(item.lower() == canonical_key for item in topics):
                matches.append(row)
                if len(matches) >= limit:
                    break
        return matches

    def subject_overview(self, subject: str, question_limit: int = 100, topic_limit: int = 20) -> dict:
        document_count = self.connection.execute(
            "SELECT COUNT(*) AS count FROM documents WHERE subject = ?",
            (subject,),
        ).fetchone()["count"]
        question_count = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
            """,
            (subject,),
        ).fetchone()["count"]
        chunk_count = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            WHERE documents.subject = ?
            """,
            (subject,),
        ).fetchone()["count"]
        return {
            "subject": subject,
            "documents": document_count,
            "chunks": chunk_count,
            "questions": question_count,
            "question_topics": self.important_topics(subject, limit=topic_limit),
            "question_rows": self.subject_questions(subject, limit=question_limit),
        }

    def stats(self) -> dict[str, int]:
        counts = {}
        for table in ("documents", "chunks", "questions"):
            counts[table] = self.connection.execute(
                f"SELECT COUNT(*) AS count FROM {table}"
            ).fetchone()["count"]
        return counts

    def subject_stats(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT subject, COUNT(*) AS document_count
            FROM documents
            GROUP BY subject
            ORDER BY subject
            """
        ).fetchall()

    def close(self) -> None:
        self.connection.close()

    def delete_subject(self, subject: str) -> dict[str, int]:
        subject = subject.strip()
        with self.connection:
            document_rows = self.connection.execute(
                "SELECT id FROM documents WHERE subject = ?",
                (subject,),
            ).fetchall()
            document_ids = [row["id"] for row in document_rows]
            deleted = {"documents": len(document_ids), "chunks": 0, "questions": 0}

            for document_id in document_ids:
                question_ids = self.connection.execute(
                    "SELECT id FROM questions WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
                deleted["questions"] += len(question_ids)
                for row in question_ids:
                    self.connection.execute(
                        "DELETE FROM questions_fts WHERE rowid = ?",
                        (row["id"],),
                    )

                chunk_ids = self.connection.execute(
                    "SELECT id FROM chunks WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
                deleted["chunks"] += len(chunk_ids)
                for row in chunk_ids:
                    self.connection.execute(
                        "DELETE FROM chunks_fts WHERE rowid = ?",
                        (row["id"],),
                    )

            self.connection.execute("DELETE FROM documents WHERE subject = ?", (subject,))
        self.connection.execute("VACUUM")
        return deleted

    def reset_all(self) -> dict[str, int]:
        counts = self.stats()
        with self.connection:
            self.connection.execute("DELETE FROM chunks_fts")
            self.connection.execute("DELETE FROM questions_fts")
            self.connection.execute("DELETE FROM questions")
            self.connection.execute("DELETE FROM chunks")
            self.connection.execute("DELETE FROM documents")
        self.connection.execute("VACUUM")
        return counts
