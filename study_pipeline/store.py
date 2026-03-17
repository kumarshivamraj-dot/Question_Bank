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

CREATE INDEX IF NOT EXISTS idx_documents_subject ON documents(subject);

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

CREATE INDEX IF NOT EXISTS idx_chunks_document_page ON chunks(document_id, page_number, id);

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
    primary_topic TEXT,
    has_diagram INTEGER NOT NULL DEFAULT 0,
    source_pdf_path TEXT,
    source_pdf_name TEXT,
    source_pdf_page INTEGER,
    original_image_path TEXT,
    diagram_image_path TEXT,
    link_confidence REAL,
    topics_json TEXT NOT NULL,
    embedding_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_questions_document_page ON questions(document_id, page_number, id);
CREATE INDEX IF NOT EXISTS idx_questions_primary_topic ON questions(primary_topic);

CREATE TABLE IF NOT EXISTS subject_topic_catalog (
    subject TEXT PRIMARY KEY,
    topics_json TEXT NOT NULL,
    syllabus_text TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS question_bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    document_path TEXT NOT NULL,
    document_name TEXT NOT NULL,
    source_type TEXT,
    page_number INTEGER NOT NULL,
    question_number TEXT,
    text TEXT NOT NULL,
    primary_topic TEXT,
    has_diagram INTEGER NOT NULL DEFAULT 0,
    source_pdf_path TEXT,
    source_pdf_name TEXT,
    source_pdf_page INTEGER,
    original_image_path TEXT,
    diagram_image_path TEXT,
    link_confidence REAL,
    topics_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_question_bookmarks_unique
ON question_bookmarks(subject, document_path, page_number, COALESCE(question_number, ''), text);
CREATE INDEX IF NOT EXISTS idx_question_bookmarks_subject ON question_bookmarks(subject, created_at DESC);

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
        self._ensure_column("questions", "primary_topic", "TEXT")
        self._ensure_column("questions", "has_diagram", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("questions", "source_pdf_path", "TEXT")
        self._ensure_column("questions", "source_pdf_name", "TEXT")
        self._ensure_column("questions", "source_pdf_page", "INTEGER")
        self._ensure_column("questions", "original_image_path", "TEXT")
        self._ensure_column("questions", "diagram_image_path", "TEXT")
        self._ensure_column("questions", "link_confidence", "REAL")
        self._ensure_column("questions", "embedding_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("question_bookmarks", "source_pdf_path", "TEXT")
        self._ensure_column("question_bookmarks", "has_diagram", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("question_bookmarks", "source_pdf_name", "TEXT")
        self._ensure_column("question_bookmarks", "source_pdf_page", "INTEGER")
        self._ensure_column("question_bookmarks", "original_image_path", "TEXT")
        self._ensure_column("question_bookmarks", "diagram_image_path", "TEXT")
        self._ensure_column("question_bookmarks", "link_confidence", "REAL")
        self.connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
                text,
                content='questions',
                content_rowid='id'
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS subject_topic_catalog (
                subject TEXT PRIMARY KEY,
                topics_json TEXT NOT NULL,
                syllabus_text TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._ensure_column("subject_topic_catalog", "syllabus_text", "TEXT NOT NULL DEFAULT ''")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS question_bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                document_path TEXT NOT NULL,
                document_name TEXT NOT NULL,
                source_type TEXT,
                page_number INTEGER NOT NULL,
                question_number TEXT,
                text TEXT NOT NULL,
                primary_topic TEXT,
                has_diagram INTEGER NOT NULL DEFAULT 0,
                source_pdf_path TEXT,
                source_pdf_name TEXT,
                source_pdf_page INTEGER,
                original_image_path TEXT,
                diagram_image_path TEXT,
                link_confidence REAL,
                topics_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_question_bookmarks_unique
            ON question_bookmarks(subject, document_path, page_number, COALESCE(question_number, ''), text)
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_question_bookmarks_subject
            ON question_bookmarks(subject, created_at DESC)
            """
        )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _fts_rowids(self, table: str, query: str, limit: int) -> list[int]:
        query = query.strip()
        if not query:
            return []
        try:
            rows = self.connection.execute(
                f"""
                SELECT rowid
                FROM {table}
                WHERE {table} MATCH ?
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(row["rowid"]) for row in rows]

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
                primary_topic = canonicalize_topic(question.primary_topic or "")
                embedding = self.embedding_provider.embed(question.text)
                cursor = self.connection.execute(
                    "INSERT INTO questions(document_id, page_number, question_number, text, primary_topic, has_diagram, source_pdf_path, source_pdf_name, source_pdf_page, original_image_path, diagram_image_path, link_confidence, topics_json, embedding_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document_id,
                        question.page_number,
                        question.question_number,
                        question.text,
                        primary_topic,
                        1 if question.has_diagram else 0,
                        question.source_pdf_path,
                        question.source_pdf_name,
                        question.source_pdf_page,
                        question.original_image_path,
                        question.diagram_image_path,
                        question.link_confidence,
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
        candidate_ids = self._fts_rowids("chunks_fts", query, max(limit * 30, 120))
        fts_scores: dict[int, float] = {}
        if candidate_ids:
            fts_scores = {
                row["rowid"]: row["keyword_score"]
                for row in self.connection.execute(
                    f"""
                    SELECT rowid, bm25(chunks_fts) AS keyword_score
                    FROM chunks_fts
                    WHERE rowid IN ({",".join("?" for _ in candidate_ids)})
                    """,
                    candidate_ids,
                ).fetchall()
            }
            rows = self.connection.execute(
                f"""
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
                WHERE chunks.id IN ({",".join("?" for _ in candidate_ids)})
                  AND (? IS NULL OR documents.subject = ?)
                """,
                (*candidate_ids, subject, subject),
            ).fetchall()
        else:
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
                ORDER BY chunks.id DESC
                LIMIT ?
                """,
                (subject, subject, max(limit * 40, 200)),
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
        candidate_ids = self._fts_rowids("questions_fts", query, max(limit * 30, 120))
        if candidate_ids:
            rows = self.connection.execute(
                f"""
                SELECT documents.name, documents.path, documents.subject, documents.source_type,
                       questions.page_number, questions.question_number, questions.text,
                       questions.primary_topic, questions.has_diagram, questions.source_pdf_path, questions.source_pdf_name,
                       questions.source_pdf_page, questions.original_image_path, questions.diagram_image_path,
                       questions.link_confidence, questions.topics_json, questions.embedding_json
                FROM questions
                JOIN documents ON documents.id = questions.document_id
                WHERE questions.id IN ({",".join("?" for _ in candidate_ids)})
                  AND (? IS NULL OR documents.subject = ?)
                """,
                (*candidate_ids, subject, subject),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT documents.name, documents.path, documents.subject, documents.source_type,
                       questions.page_number, questions.question_number, questions.text,
                       questions.primary_topic, questions.has_diagram, questions.source_pdf_path, questions.source_pdf_name,
                       questions.source_pdf_page, questions.original_image_path, questions.diagram_image_path,
                       questions.link_confidence, questions.topics_json, questions.embedding_json
                FROM questions
                JOIN documents ON documents.id = questions.document_id
                WHERE (? IS NULL OR documents.subject = ?)
                ORDER BY questions.id DESC
                LIMIT ?
                """,
                (subject, subject, max(limit * 40, 200)),
            ).fetchall()

        scored = []
        query_lower = query.lower()
        for row in rows:
            vector_score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            topics = [topic.lower() for topic in json.loads(row["topics_json"])]
            keyword_bonus = 0.2 if query_lower in row["text"].lower() else 0.0
            primary_topic = canonicalize_topic(row["primary_topic"] or "")
            if primary_topic and query_lower in primary_topic.lower():
                keyword_bonus += 0.35
            if any(query_lower in topic for topic in topics):
                keyword_bonus += 0.3
            scored.append((vector_score + keyword_bonus, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def subject_topics(self, subject: str, limit: int = 15) -> list[tuple[str, int]]:
        rows = self.connection.execute(
            """
            SELECT questions.primary_topic, COUNT(*) AS question_count
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
              AND questions.primary_topic IS NOT NULL
              AND TRIM(questions.primary_topic) != ''
            GROUP BY questions.primary_topic
            ORDER BY question_count DESC, LOWER(questions.primary_topic) ASC
            LIMIT ?
            """,
            (subject, limit),
        ).fetchall()
        return [
            (topic, int(row["question_count"]))
            for row in rows
            if (topic := canonicalize_topic(row["primary_topic"] or ""))
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
        topic_rows = self.connection.execute(
            """
            SELECT questions.primary_topic,
                   COUNT(*) AS question_count,
                   SUM(
                       2.0 + CASE
                           WHEN INSTR(LOWER(questions.text), LOWER(questions.primary_topic)) > 0 THEN 0.35
                           ELSE 0.0
                       END
                   ) AS score
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
              AND questions.primary_topic IS NOT NULL
              AND TRIM(questions.primary_topic) != ''
            GROUP BY questions.primary_topic
            ORDER BY question_count DESC, score DESC, LOWER(questions.primary_topic) ASC
            LIMIT ?
            """,
            (subject, limit),
        ).fetchall()
        ranked_topics = [
            {
                "topic": topic,
                "score": round(float(row["score"]), 2),
                "question_count": int(row["question_count"]),
                "chunk_count": 0,
                "examples": [],
            }
            for row in topic_rows
            if (topic := canonicalize_topic(row["primary_topic"] or ""))
        ]
        for item in ranked_topics:
            example_rows = self.connection.execute(
                """
                SELECT questions.text
                FROM questions
                JOIN documents ON documents.id = questions.document_id
                WHERE documents.subject = ?
                  AND questions.primary_topic = ?
                ORDER BY questions.id DESC
                LIMIT 2
                """,
                (subject, item["topic"]),
            ).fetchall()
            item["examples"] = [str(row["text"])[:180].strip() for row in example_rows]
        return ranked_topics

    def subject_questions(self, subject: str, limit: int = 100) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT documents.name, documents.path, documents.subject, documents.source_type,
                   questions.page_number, questions.question_number, questions.text,
                   questions.primary_topic, questions.has_diagram, questions.source_pdf_path, questions.source_pdf_name,
                   questions.source_pdf_page, questions.original_image_path, questions.diagram_image_path,
                   questions.link_confidence, questions.topics_json
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

        return self.connection.execute(
            """
            SELECT documents.name, documents.path, documents.subject, documents.source_type,
                   questions.page_number, questions.question_number, questions.text,
                   questions.primary_topic, questions.has_diagram, questions.source_pdf_path, questions.source_pdf_name,
                   questions.source_pdf_page, questions.original_image_path, questions.diagram_image_path,
                   questions.link_confidence, questions.topics_json
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
              AND questions.primary_topic = ?
            ORDER BY documents.name, questions.page_number, questions.id
            LIMIT ?
            """,
            (subject, canonical_topic, limit),
        ).fetchall()

    def subject_question_texts(self, subject: str, limit: int = 300) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT questions.text
            FROM questions
            JOIN documents ON documents.id = questions.document_id
            WHERE documents.subject = ?
            ORDER BY questions.id DESC
            LIMIT ?
            """,
            (subject, limit),
        ).fetchall()
        return [str(row["text"]) for row in rows]

    def get_subject_topic_catalog(self, subject: str) -> list[str]:
        row = self.connection.execute(
            "SELECT topics_json FROM subject_topic_catalog WHERE subject = ?",
            (subject,),
        ).fetchone()
        if not row:
            return []
        return canonicalize_topics(json.loads(row["topics_json"]))

    def get_subject_syllabus(self, subject: str) -> str:
        row = self.connection.execute(
            "SELECT syllabus_text FROM subject_topic_catalog WHERE subject = ?",
            (subject,),
        ).fetchone()
        return str(row["syllabus_text"]) if row and row["syllabus_text"] is not None else ""

    def set_subject_topic_catalog(self, subject: str, topics: list[str]) -> None:
        normalized_topics = canonicalize_topics(topics)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO subject_topic_catalog(subject, topics_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(subject) DO UPDATE SET
                topics_json = excluded.topics_json,
                updated_at = CURRENT_TIMESTAMP
                """,
                (subject, json.dumps(normalized_topics)),
            )

    def set_subject_syllabus(self, subject: str, syllabus_text: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO subject_topic_catalog(subject, topics_json, syllabus_text, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(subject) DO UPDATE SET
                syllabus_text = excluded.syllabus_text,
                updated_at = CURRENT_TIMESTAMP
                """,
                (subject, json.dumps(self.get_subject_topic_catalog(subject)), syllabus_text.strip()),
            )

    def add_bookmark(
        self,
        *,
        subject: str,
        document_path: str,
        document_name: str,
        source_type: str | None,
        page_number: int,
        question_number: str | None,
        text: str,
        primary_topic: str | None,
        has_diagram: bool,
        source_pdf_path: str | None,
        source_pdf_name: str | None,
        source_pdf_page: int | None,
        original_image_path: str | None,
        diagram_image_path: str | None,
        link_confidence: float | None,
        topics: list[str],
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO question_bookmarks(
                    subject, document_path, document_name, source_type,
                    page_number, question_number, text, primary_topic, has_diagram,
                    source_pdf_path, source_pdf_name, source_pdf_page,
                    original_image_path, diagram_image_path, link_confidence,
                    topics_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject.strip(),
                    document_path,
                    document_name,
                    source_type,
                    int(page_number),
                    question_number,
                    text.strip(),
                    canonicalize_topic(primary_topic or ""),
                    1 if has_diagram else 0,
                    source_pdf_path,
                    source_pdf_name,
                    source_pdf_page,
                    original_image_path,
                    diagram_image_path,
                    link_confidence,
                    json.dumps(canonicalize_topics(topics)),
                ),
            )

    def remove_bookmark(
        self,
        *,
        subject: str,
        document_path: str,
        page_number: int,
        question_number: str | None,
        text: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                DELETE FROM question_bookmarks
                WHERE subject = ?
                  AND document_path = ?
                  AND page_number = ?
                  AND COALESCE(question_number, '') = COALESCE(?, '')
                  AND text = ?
                """,
                (subject.strip(), document_path, int(page_number), question_number, text.strip()),
            )

    def bookmark_exists(
        self,
        *,
        subject: str,
        document_path: str,
        page_number: int,
        question_number: str | None,
        text: str,
    ) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM question_bookmarks
            WHERE subject = ?
              AND document_path = ?
              AND page_number = ?
              AND COALESCE(question_number, '') = COALESCE(?, '')
              AND text = ?
            LIMIT 1
            """,
            (subject.strip(), document_path, int(page_number), question_number, text.strip()),
        ).fetchone()
        return bool(row)

    def bookmarks(self, subject: str, limit: int = 200) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT subject, document_path, document_name, source_type, page_number,
                   question_number, text, primary_topic, has_diagram, source_pdf_path, source_pdf_name,
                   source_pdf_page, original_image_path, diagram_image_path, link_confidence,
                   topics_json, created_at
            FROM question_bookmarks
            WHERE subject = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (subject.strip(), limit),
        ).fetchall()

    def subject_overview(
        self,
        subject: str,
        question_limit: int = 100,
        topic_limit: int = 20,
        include_questions: bool = False,
    ) -> dict:
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
            "question_rows": self.subject_questions(subject, limit=question_limit) if include_questions else [],
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
