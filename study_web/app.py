from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from study_pipeline.embeddings import HashingEmbeddingProvider, OllamaEmbeddingProvider
from study_pipeline.store import StudyStore
from study_pipeline.topics import canonicalize_topics


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = Path(os.environ.get("STUDY_UPLOAD_ROOT", BASE_DIR / "data" / "uploads"))
DB_PATH = Path(os.environ.get("STUDY_DB_PATH", BASE_DIR / "data" / "processed" / "study_index.db"))
EMBEDDING_PROVIDER = os.environ.get("STUDY_EMBEDDING_PROVIDER", "hashing")
OLLAMA_MODEL = os.environ.get("STUDY_OLLAMA_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("STUDY_OLLAMA_URL", "http://localhost:11434")
TOPIC_OLLAMA_MODEL = os.environ.get("STUDY_TOPIC_OLLAMA_MODEL", "llama3")
TOPIC_OLLAMA_URL = os.environ.get("STUDY_TOPIC_OLLAMA_URL", OLLAMA_URL)
ORIGINAL_ASSET_ROOT = Path(os.environ.get("STUDY_ORIGINAL_ASSET_ROOT", BASE_DIR / "data" / "processed" / "original_views"))
ADMIN_KEY = os.environ.get("STUDY_ADMIN_KEY", "").strip()

app = FastAPI(title="Study Index")


def build_embedding_provider():
    if EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddingProvider(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
    return HashingEmbeddingProvider()


def get_store() -> StudyStore:
    return StudyStore(DB_PATH, build_embedding_provider())


def safe_filename(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return cleaned or "upload.bin"


def asset_url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    try:
        relative = resolved.relative_to(ORIGINAL_ASSET_ROOT.resolve())
    except ValueError:
        return None
    return f"/api/assets/original?path={quote(relative.as_posix())}"


def question_payload(row) -> dict:
    original_image_path = row["original_image_path"]
    diagram_image_path = row["diagram_image_path"]
    source_pdf_name = row["source_pdf_name"]
    if not original_image_path and row["source_pdf_page"]:
        from study_pipeline.pdf_linker import materialize_original_asset

        original_image_path, diagram_image_path, source_pdf_name = materialize_original_asset(
            subject=row["subject"],
            document_path=row["path"],
            question_number=row["question_number"],
            text=row["text"],
            has_diagram=bool(row["has_diagram"]),
            source_pdf_name=row["source_pdf_name"],
            source_pdf_page=row["source_pdf_page"],
        )
    return {
        "subject": row["subject"],
        "source_type": row["source_type"],
        "document": row["name"],
        "page": row["page_number"],
        "question_number": row["question_number"],
        "primary_topic": row["primary_topic"],
        "has_diagram": bool(row["has_diagram"]),
        "topics": canonicalize_topics(json.loads(row["topics_json"])),
        "path": row["path"],
        "text": row["text"],
        "source_pdf_path": row["source_pdf_path"],
        "source_pdf_name": source_pdf_name,
        "source_pdf_page": row["source_pdf_page"],
        "original_image_path": original_image_path,
        "diagram_image_path": diagram_image_path,
        "original_view_url": asset_url(original_image_path),
        "diagram_url": asset_url(diagram_image_path),
        "link_confidence": row["link_confidence"],
    }


def save_upload(subject: str, source_type: str, file: UploadFile) -> Path:
    subject_dir = UPLOAD_ROOT / subject / source_type
    subject_dir.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(file.filename or "upload.bin")
    destination = subject_dir / filename
    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while destination.exists():
        destination = subject_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return destination


def verify_admin(x_admin_key: str | None) -> None:
    if not ADMIN_KEY:
        return
    if (x_admin_key or "").strip() != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin key required")


def ingest_path(path: Path, subject: str, source_type: str, vision_provider: str, ocr_lang: str, chunk_size: int) -> dict:
    from study_pipeline.chunking import chunk_page
    from study_pipeline.extract import extract_document
    from study_pipeline.json_input import load_question_json
    from study_pipeline.pdf_linker import link_questions_to_pdf
    from study_pipeline.questions import extract_questions
    from study_pipeline.topic_classifier import infer_primary_topics
    from study_pipeline.vision import extract_pyq_questions_with_vision_fallback

    if path.suffix.lower() == ".json":
        pages, questions = load_question_json(path)
        questions = link_questions_to_pdf(subject, path, questions)
    else:
        pages = extract_document(path, ocr_lang=ocr_lang)
        questions = []
        for page in pages:
            if source_type == "pyq":
                questions.extend(
                    extract_questions(
                        page.document_path,
                        page.document_name,
                        page.page_number,
                        page.text,
                    )
                )
        if source_type == "pyq" and path.suffix.lower() == ".pdf":
            questions = extract_pyq_questions_with_vision_fallback(path, questions, vision_provider)

    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page, target_size=chunk_size))

    store = get_store()
    try:
        if questions:
            existing_catalog = store.get_subject_topic_catalog(subject)
            existing_texts = store.subject_question_texts(subject)
            syllabus_text = store.get_subject_syllabus(subject)
            catalog, questions = infer_primary_topics(
                subject=subject,
                questions=questions,
                existing_question_texts=existing_texts,
                existing_topics=existing_catalog,
                syllabus_text=syllabus_text,
                base_url=TOPIC_OLLAMA_URL,
                model=TOPIC_OLLAMA_MODEL,
            )
            if catalog:
                store.set_subject_topic_catalog(subject, catalog)
        store.upsert_document(
            path=str(path),
            name=path.name,
            subject=subject,
            kind=path.suffix.lower().lstrip("."),
            source_type=source_type,
            chunks=chunks,
            questions=questions,
        )
    finally:
        store.close()
    return {
        "subject": subject,
        "source_type": source_type,
        "document": path.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "questions": len(questions),
        "saved_path": str(path),
    }


def remove_subject_uploads(subject: str) -> None:
    subject_dir = UPLOAD_ROOT / subject
    if subject_dir.exists():
        shutil.rmtree(subject_dir)


def remove_all_uploads() -> None:
    if UPLOAD_ROOT.exists():
        shutil.rmtree(UPLOAD_ROOT)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def remove_database_file() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    admin_protected = "true" if bool(ADMIN_KEY) else "false"
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Study Index</title>
  <style>
    :root {
      --ink: #1c261c;
      --muted: #627061;
      --line: #d8ded1;
      --panel: rgba(252, 253, 248, 0.92);
      --panel-strong: #fffffc;
      --accent: #244c3b;
      --accent-soft: #e4ede4;
      --accent-warm: #8c6a3c;
      --canvas: #f4f6ef;
    }
    body.dark-mode {
      --ink: #e6efe4;
      --muted: #9cb09d;
      --line: #34423b;
      --panel: rgba(20, 28, 24, 0.9);
      --panel-strong: #18221d;
      --accent: #8db39d;
      --accent-soft: rgba(141, 179, 157, 0.16);
      --accent-warm: #b28c58;
      --canvas: #0f1512;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.45;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(36, 76, 59, 0.08), transparent 24%),
        linear-gradient(180deg, #f7f8f3 0%, #eef1e6 100%);
    }
    body.dark-mode {
      background:
        radial-gradient(circle at top left, rgba(141, 179, 157, 0.12), transparent 24%),
        linear-gradient(180deg, #141b17 0%, #0e1411 100%);
    }
    .shell {
      max-width: 1040px;
      margin: 0 auto;
      padding: 18px 14px 28px;
    }
    .topbar {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 8px;
    }
    .hero {
      display: grid;
      gap: 4px;
      margin-bottom: 12px;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(1.8rem, 3.8vw, 3.1rem);
      line-height: 0.96;
      letter-spacing: -0.025em;
    }
    .hero p {
      margin: 0;
      max-width: 700px;
      color: var(--muted);
      font-size: 0.96rem;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 8px 18px rgba(28, 38, 28, 0.04);
    }
    body.dark-mode .panel,
    body.dark-mode .result-card,
    body.dark-mode .topic-sidebar {
      background: rgba(20, 28, 24, 0.94);
      border-color: #34423b;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
    }
    .panel h2 {
      margin: 0;
      font-size: 0.98rem;
    }
    .panel-note {
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.4;
    }
    .subject-bar, .stack, .field, .results, .status, .panel-head, .subject-summary {
      display: grid;
      gap: 8px;
    }
    .subject-row, .toolbar, .tab-row, .topic-row, .result-meta, .subject-topline, .subject-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .mode-switch {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .theme-toggle {
      width: 44px;
      height: 44px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      border-radius: 999px;
      background: var(--panel);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: 0 8px 18px rgba(28, 38, 28, 0.06);
      font-size: 1.05rem;
    }
    body.dark-mode .theme-toggle {
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.26);
    }
    body.dark-mode .result-text,
    body.dark-mode .question-card .result-text,
    body.dark-mode .panel h2,
    body.dark-mode .hero h1,
    body.dark-mode .figure-label {
      color: #f4f8f2;
    }
    body.dark-mode .panel-note,
    body.dark-mode .result-meta,
    body.dark-mode .counter,
    body.dark-mode .original-meta,
    body.dark-mode .hero p,
    body.dark-mode .eyebrow,
    body.dark-mode label {
      color: #b8c7ba;
    }
    .subjectDashboard {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }
    .subject-summary {
      grid-template-columns: repeat(2, minmax(0, max-content));
      gap: 8px;
    }
    .question-shell {
      display: grid;
      gap: 10px;
      justify-items: center;
    }
    .question-stage {
      width: min(760px, 100%);
      display: grid;
      gap: 10px;
    }
    .topic-toggle-row {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
    }
    .sidebar-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(28, 38, 28, 0.22);
      opacity: 0;
      pointer-events: none;
      transition: opacity 160ms ease;
      z-index: 20;
    }
    .sidebar-backdrop.visible {
      opacity: 1;
      pointer-events: auto;
    }
    .topic-sidebar {
      position: fixed;
      top: 0;
      left: 0;
      height: 100vh;
      width: min(360px, 88vw);
      padding: 16px 14px;
      background: rgba(255,255,252,0.98);
      border-right: 1px solid var(--line);
      box-shadow: 0 18px 40px rgba(28, 38, 28, 0.16);
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 10px;
      transform: translateX(-104%);
      transition: transform 180ms ease;
      z-index: 30;
    }
    .topic-sidebar.open {
      transform: translateX(0);
    }
    .sidebar-head {
      display: grid;
      gap: 8px;
    }
    .sidebar-topline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .sidebar-switch {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .sidebar-switch .tab {
      padding: 7px 11px;
    }
    .meta-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 700;
    }
    .eyebrow {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }
    label {
      font-size: 0.72rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    input, select, textarea, button { font: inherit; }
    input, select, textarea {
      width: 100%;
      padding: 9px 11px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
    }
    body.dark-mode input,
    body.dark-mode select,
    body.dark-mode textarea {
      background: #121916;
      color: #f4f8f2;
      border-color: #34423b;
    }
    .subject-row input { max-width: 240px; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 9px 14px;
      color: white;
      background: var(--accent);
      cursor: pointer;
      font-weight: 700;
    }
    button.secondary {
      background: var(--accent-warm);
    }
    .tab {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.84);
      color: var(--ink);
      cursor: pointer;
      font-weight: 700;
    }
    .tab.active {
      background: var(--accent);
      color: white;
      border-color: transparent;
    }
    body.dark-mode .tab {
      background: rgba(18, 25, 22, 0.88);
      color: #e6efe4;
      border-color: #34423b;
    }
    .dropzone {
      border: 1px dashed rgba(36, 76, 59, 0.35);
      border-radius: 12px;
      padding: 14px;
      min-height: 88px;
      display: grid;
      place-items: center;
      text-align: center;
      background: rgba(255,255,255,0.55);
      transition: 160ms ease;
    }
    body.dark-mode .dropzone {
      background: rgba(18, 25, 22, 0.76);
      border-color: rgba(141, 179, 157, 0.28);
    }
    .dropzone.dragover {
      border-color: var(--accent);
      background: rgba(228, 237, 228, 0.86);
    }
    .dropzone strong {
      display: block;
      font-size: 0.96rem;
      margin-bottom: 4px;
    }
    .files {
      margin-top: 8px;
      display: grid;
      gap: 6px;
      font-size: 0.88rem;
    }
    .file-item {
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.88);
      border: 1px solid var(--line);
    }
    .results, .status {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      min-height: 120px;
      overflow: auto;
    }
    .topic-results {
      min-height: 0;
    }
    .status {
      white-space: pre-wrap;
      font-size: 0.88rem;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
    .result-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,249,243,0.9));
      padding: 12px;
      display: grid;
      gap: 8px;
    }
    body.dark-mode .result-card {
      background: linear-gradient(180deg, rgba(24, 34, 29, 0.98), rgba(16, 23, 20, 0.96));
    }
    .question-viewer {
      display: grid;
      gap: 10px;
    }
    .question-frame {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 10px;
      align-items: start;
    }
    .copy-rail {
      display: grid;
      align-content: start;
      gap: 8px;
    }
    .copy-question-btn {
      min-width: 0;
      padding: 10px 12px;
      border-radius: 12px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid rgba(36, 76, 59, 0.14);
    }
    .copy-question-btn.copied {
      background: var(--accent);
      color: white;
    }
    .bookmark-icon-btn {
      width: 44px;
      height: 44px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      font-size: 1.05rem;
      line-height: 1;
    }
    .copy-question-btn:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .question-nav {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }
    .question-nav .toolbar {
      gap: 6px;
    }
    .question-panel {
      min-height: 280px;
    }
    .counter {
      font-size: 0.8rem;
      color: var(--muted);
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .topic-card {
      width: 100%;
      text-align: left;
      color: var(--ink);
      background: transparent;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      display: grid;
      gap: 6px;
    }
    .topic-card.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .result-meta {
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .chip, .topic-pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 9px;
      font-weight: 700;
    }
    .chip {
      background: var(--accent-soft);
      color: var(--accent);
    }
    .topic-pill {
      background: rgba(36, 76, 59, 0.1);
      color: var(--accent);
      font-size: 0.76rem;
    }
    .result-text {
      white-space: pre-wrap;
      line-height: 1.42;
      font-size: 0.94rem;
    }
    .question-card {
      min-height: 220px;
      align-content: center;
      justify-items: center;
      text-align: center;
    }
    body.dark-mode .question-card {
      background: linear-gradient(180deg, rgba(22, 31, 27, 0.99), rgba(10, 15, 13, 0.98));
    }
    .question-card .result-meta,
    .question-card .result-text,
    .question-card .topic-row {
      justify-content: center;
      text-align: center;
    }
    .question-card .result-text {
      max-width: 62ch;
      margin: 0 auto;
      font-size: 1rem;
      line-height: 1.55;
    }
    .original-panel {
      display: grid;
      gap: 10px;
      text-align: left;
      justify-items: stretch;
    }
    .original-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 0.8rem;
    }
    .original-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    .original-figure {
      display: grid;
      gap: 6px;
    }
    .original-figure img {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }
    .figure-label {
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .result-path {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.74rem;
      color: var(--muted);
      word-break: break-all;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .workflow-steps {
      display: grid;
      gap: 8px;
    }
    .workflow-step {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.5);
      display: grid;
      gap: 4px;
    }
    body.dark-mode .workflow-step {
      background: rgba(18, 25, 22, 0.72);
    }
    .workflow-kicker {
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }
    .single-column {
      grid-column: 1 / -1;
    }
    .hidden { display: none; }
    @media (max-width: 920px) {
      .mini-grid, .subject-summary { grid-template-columns: 1fr; }
      .question-stage {
        width: 100%;
      }
      .question-nav {
        justify-content: center;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <button id="themeToggleBtn" class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">☾</button>
    </div>
    <section class="hero">
      <div class="eyebrow">PYQ Topic Navigator</div>
      <h1>Pick a subject. Open a topic. See the actual questions.</h1>
      <p>The UI stays focused on the PYQ topics that matter most. Subjects open into an importance-ranked topic list, and clicking a topic shows the matching extracted questions.</p>
    </section>

    <section class="panel subject-bar">
      <div class="subject-row">
        <div class="field">
          <label for="subjectInput">Subject Name</label>
          <input id="subjectInput" placeholder="dbms">
        </div>
        <button id="saveSubjectBtn">Open Subject</button>
        <div id="subjectHint" class="panel-note">Create or select a subject tab to begin.</div>
      </div>
      <div id="subjectTabs" class="tab-row"></div>
    </section>

    <section class="panel">
      <div class="mode-switch">
        <button id="practiceModeBtn" class="tab active" type="button">Practice View</button>
        <button id="adminModeBtn" class="tab" type="button">Admin Panel</button>
      </div>
    </section>

    <section id="emptyState" class="panel stack">
      <h2>No Subject Open</h2>
      <div class="panel-note">Open an existing subject or create a new one, then follow the workflow: syllabus first, cleaned JSON second, original PDF third.</div>
    </section>

    <section id="subjectDashboard" class="hidden">
      <section class="panel stack">
        <div class="subject-topline">
          <div class="stack">
            <div class="eyebrow">Current Subject</div>
            <h2 id="currentSubjectLabel">-</h2>
          </div>
          <div class="subject-summary">
            <div class="meta-pill">Questions <span id="overviewQuestions">0</span></div>
            <div class="meta-pill">Documents <span id="overviewDocuments">0</span></div>
          </div>
        </div>
        <div class="panel-note">Topics are ordered from most important to least important based on extracted PYQ frequency.</div>
      </section>

      <section id="adminView" class="hidden">
        <section id="ingestPanel" class="panel stack">
          <div class="panel-head">
            <h2>Subject Workflow</h2>
            <div class="panel-note">Set the syllabus first, then upload the cleaned question JSON, then upload the original PDF for page-based original view.</div>
          </div>
          <div class="workflow-steps">
            <div class="workflow-step">
              <div class="workflow-kicker">Step 1</div>
              <strong>Paste and save the syllabus.</strong>
              <div class="panel-note">These syllabus topics become the allowed topic catalog for this subject.</div>
            </div>
            <div class="workflow-step">
              <div class="workflow-kicker">Step 2</div>
              <strong>Upload the cleaned JSON of questions.</strong>
              <div class="panel-note">The JSON should already contain clean question text and, when needed, the mapped `source_pdf_name` and `source_pdf_page`.</div>
            </div>
            <div class="workflow-step">
              <div class="workflow-kicker">Step 3</div>
              <strong>Upload the original PYQ PDF.</strong>
              <div class="panel-note">The `Original` button will render the mapped page from this PDF.</div>
            </div>
          </div>
          <div class="mini-grid">
            <div class="field">
              <label for="sourceType">Source Type</label>
              <select id="sourceType">
                <option value="pyq">PYQ</option>
                <option value="reference">Reference</option>
                <option value="handout">Handout</option>
                <option value="notes">Notes</option>
                <option value="ppt">PPT</option>
              </select>
            </div>
            <div class="field">
              <label for="visionProvider">Vision For PYQs</label>
              <select id="visionProvider">
                <option value="none">None</option>
                <option value="claude_cheap">Claude Cheap Fallback</option>
                <option value="claude">Claude</option>
              </select>
            </div>
            <div class="field single-column">
              <label for="ocrLang">OCR Language</label>
              <input id="ocrLang" value="eng">
            </div>
          </div>
          <div id="dropzone" class="dropzone">
            <div>
              <strong>Upload one stage at a time</strong>
              <div>First upload the cleaned JSON. Then upload the original PDF in a separate upload.</div>
              <input id="fileInput" type="file" multiple hidden accept=".pdf,.json,.pptx,.txt,.md">
              <div id="fileList" class="files"></div>
            </div>
          </div>
          <div class="toolbar">
            <button id="uploadBtn">Ingest Files</button>
            <button id="saveSyllabusBtn" class="secondary">Save Syllabus</button>
            <button id="refreshSubjectBtn" class="secondary">Refresh Subject</button>
            <button id="deleteSubjectBtn" class="secondary">Delete This Subject</button>
            <button id="resetDbBtn" class="secondary">Delete Entire Database</button>
          </div>
          <div class="field">
            <label for="syllabusInput">Syllabus Topics</label>
            <textarea id="syllabusInput" rows="8" placeholder="Paste the syllabus here. Use one topic per line for the cleanest mapping."></textarea>
          </div>
          <div class="field">
            <label>Status</label>
            <div id="status" class="status">No uploads yet.</div>
          </div>
        </section>
      </section>

      <section id="practiceView">
        <div id="sidebarBackdrop" class="sidebar-backdrop"></div>
        <aside id="topicSidebar" class="topic-sidebar" aria-hidden="true">
          <div class="sidebar-head">
            <div class="sidebar-topline">
              <div class="stack">
                <h2 id="sidebarTitle">Important Topics</h2>
                <div id="sidebarHint" class="panel-note">Most asked first. Click a topic to load the matching PYQ questions.</div>
              </div>
              <button id="closeSidebarBtn" class="secondary" type="button">Close</button>
            </div>
            <div class="sidebar-switch">
              <button id="sidebarTopicsBtn" class="tab active" type="button">Topics</button>
              <button id="sidebarBookmarksBtn" class="tab" type="button">Bookmarks</button>
            </div>
          </div>
          <div id="topicResults" class="results topic-results">No topics yet.</div>
        </aside>

        <section class="question-shell">
          <div class="topic-toggle-row">
            <button id="openSidebarBtn" type="button">Important Topics</button>
          </div>
          <div class="panel question-stage">
            <div class="panel-head">
              <div class="stack">
                <h2 id="selectedTopicLabel">Questions</h2>
                <div id="questionHint" class="panel-note">Choose a topic from the left.</div>
              </div>
            </div>
            <div class="question-nav">
              <div id="questionCounter" class="counter">0 questions</div>
              <div class="toolbar">
                <button id="prevQuestionBtn" class="secondary" type="button">Previous</button>
                <button id="nextQuestionBtn" type="button">Next</button>
              </div>
            </div>
            <div id="topicQuestions" class="results question-panel">No questions yet.</div>
          </div>
        </section>
      </section>
    </section>
  </div>

  <script>
    const emptyState = document.getElementById('emptyState');
    const subjectDashboard = document.getElementById('subjectDashboard');
    const practiceView = document.getElementById('practiceView');
    const adminView = document.getElementById('adminView');
    const subjectInput = document.getElementById('subjectInput');
    const subjectTabs = document.getElementById('subjectTabs');
    const subjectHint = document.getElementById('subjectHint');
    const currentSubjectLabel = document.getElementById('currentSubjectLabel');
    const overviewQuestions = document.getElementById('overviewQuestions');
    const overviewDocuments = document.getElementById('overviewDocuments');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const syllabusInput = document.getElementById('syllabusInput');
    const statusBox = document.getElementById('status');
    const topicResults = document.getElementById('topicResults');
    const topicQuestions = document.getElementById('topicQuestions');
    const selectedTopicLabel = document.getElementById('selectedTopicLabel');
    const questionHint = document.getElementById('questionHint');
    const questionCounter = document.getElementById('questionCounter');
    const prevQuestionBtn = document.getElementById('prevQuestionBtn');
    const nextQuestionBtn = document.getElementById('nextQuestionBtn');
    const topicSidebar = document.getElementById('topicSidebar');
    const sidebarTitle = document.getElementById('sidebarTitle');
    const sidebarHint = document.getElementById('sidebarHint');
    const sidebarTopicsBtn = document.getElementById('sidebarTopicsBtn');
    const sidebarBookmarksBtn = document.getElementById('sidebarBookmarksBtn');
    const sidebarBackdrop = document.getElementById('sidebarBackdrop');
    const openSidebarBtn = document.getElementById('openSidebarBtn');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const practiceModeBtn = document.getElementById('practiceModeBtn');
    const adminModeBtn = document.getElementById('adminModeBtn');
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const adminProtected = __ADMIN_PROTECTED__;
    let selectedFiles = [];
    let knownSubjects = [];
    let currentSubject = localStorage.getItem('study.currentSubject') || '';
    let currentTopic = '';
    let activeMode = localStorage.getItem('study.activeMode') || 'practice';
    let activeTheme = localStorage.getItem('study.theme') || 'light';
    let adminKey = sessionStorage.getItem('study.adminKey') || '';
    let latestSubjectOverview = null;
    let currentTopicQuestions = [];
    let currentQuestionIndex = 0;
    let showingBookmarks = false;
    let sidebarMode = 'topics';
    let sidebarOpen = false;

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function renderEmpty(message) {
      return `<div class="empty">${escapeHtml(message)}</div>`;
    }

    function renderTopics(topics) {
      if (!topics || !topics.length) return '';
      return `<div class="topic-row">${topics.map(topic => `<span class="topic-pill">${escapeHtml(topic)}</span>`).join('')}</div>`;
    }

    function syncSidebarMode() {
      const bookmarksMode = sidebarMode === 'bookmarks';
      sidebarTopicsBtn.classList.toggle('active', !bookmarksMode);
      sidebarBookmarksBtn.classList.toggle('active', bookmarksMode);
      sidebarTitle.textContent = bookmarksMode ? 'Bookmarked Questions' : 'Important Topics';
      sidebarHint.textContent = bookmarksMode
        ? 'Saved questions for this subject. Click one to open it in the viewer.'
        : 'Most asked first. Click a topic to load the matching PYQ questions.';
    }

    function renderOriginalPanel(item) {
      if (!item || !item.showOriginal || !item.original_view_url) {
        return '';
      }
      const source = item.source_pdf_name
        ? `${escapeHtml(item.source_pdf_name)} page ${escapeHtml(item.source_pdf_page ?? '-')}`
        : 'Linked PDF source';
      const confidence = item.link_confidence != null
        ? `<span>match ${escapeHtml((Number(item.link_confidence) * 100).toFixed(0))}%</span>`
        : '';
      return `
        <section class="result-card original-panel">
          <div class="original-meta">
            <span>${source}</span>
            ${confidence}
          </div>
          <div class="original-grid">
            <figure class="original-figure">
              <figcaption class="figure-label">Original View</figcaption>
              <img src="${escapeHtml(item.original_view_url)}" alt="Original question crop">
            </figure>
            ${item.diagram_url ? `
              <figure class="original-figure">
                <figcaption class="figure-label">Detected Diagram</figcaption>
                <img src="${escapeHtml(item.diagram_url)}" alt="Detected diagram crop">
              </figure>
            ` : ''}
          </div>
        </section>
      `;
    }

    function renderQuestionViewer(items, index) {
      if (!items || !items.length) {
        return renderEmpty('No extracted questions for this topic yet.');
      }
      const safeIndex = Math.min(Math.max(index, 0), items.length - 1);
      const item = items[safeIndex];
      return `
        <div class="question-viewer">
          <div class="question-frame">
            <div class="copy-rail">
              <button class="copy-question-btn" type="button" data-copy-question="true" aria-label="Copy question text">Copy</button>
              <button class="copy-question-btn" type="button" data-toggle-original="true" ${item.original_view_url ? '' : 'disabled'} aria-label="Toggle original source">${item.showOriginal ? 'Hide' : 'Original'}</button>
              <button class="copy-question-btn bookmark-icon-btn ${item.bookmarked ? 'copied' : ''}" type="button" data-bookmark-question="true" aria-label="Bookmark question" title="${item.bookmarked ? 'Remove bookmark' : 'Add bookmark'}">${item.bookmarked ? '★' : '☆'}</button>
            </div>
            <article class="result-card question-card">
              <div class="result-meta">
                <span class="chip">${escapeHtml(item.source_type || 'pyq')}</span>
                <span>${escapeHtml(item.document || '-')}</span>
                <span>page ${escapeHtml(item.page ?? '-')}</span>
                ${item.question_number ? `<span class="chip">${escapeHtml(item.question_number)}</span>` : ''}
                ${item.has_diagram ? `<span class="chip">[Diagram]</span>` : ''}
              </div>
              <div class="result-text">${escapeHtml(item.text || '')}</div>
              ${renderTopics(item.topics || [])}
            </article>
          </div>
          ${renderOriginalPanel(item)}
        </div>
      `;
    }

    async function copyCurrentQuestion(button) {
      const item = currentTopicQuestions[currentQuestionIndex];
      if (!item || !item.text) {
        return;
      }
      try {
        await navigator.clipboard.writeText(String(item.text));
        const original = button.textContent;
        button.textContent = 'Copied';
        button.classList.add('copied');
        window.setTimeout(() => {
          button.textContent = original;
          button.classList.remove('copied');
        }, 1200);
      } catch (error) {
        window.alert('Copy failed. Your browser blocked clipboard access.');
      }
    }

    async function toggleBookmarkCurrentQuestion(button) {
      const item = currentTopicQuestions[currentQuestionIndex];
      if (!item || !currentSubject) {
        return;
      }
      const method = item.bookmarked ? 'DELETE' : 'POST';
      const response = await fetch('/api/bookmarks', {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject: currentSubject,
          path: item.path,
          document: item.document,
          source_type: item.source_type,
          page: item.page,
          question_number: item.question_number,
          text: item.text,
          primary_topic: item.primary_topic,
          has_diagram: Boolean(item.has_diagram),
          source_pdf_path: item.source_pdf_path,
          source_pdf_name: item.source_pdf_name,
          source_pdf_page: item.source_pdf_page,
          original_image_path: item.original_image_path,
          diagram_image_path: item.diagram_image_path,
          link_confidence: item.link_confidence,
          topics: item.topics || [],
        }),
      });
      if (!response.ok) {
        window.alert('Could not update bookmark.');
        return;
      }
      item.bookmarked = !item.bookmarked;
      button.textContent = item.bookmarked ? '★' : '☆';
      button.setAttribute('title', item.bookmarked ? 'Remove bookmark' : 'Add bookmark');
      button.classList.toggle('copied', item.bookmarked);
      if (showingBookmarks && !item.bookmarked) {
        currentTopicQuestions.splice(currentQuestionIndex, 1);
        if (currentQuestionIndex >= currentTopicQuestions.length) {
          currentQuestionIndex = Math.max(0, currentTopicQuestions.length - 1);
        }
        renderCurrentQuestion();
      }
    }

    function toggleOriginalCurrentQuestion() {
      const item = currentTopicQuestions[currentQuestionIndex];
      if (!item || !item.original_view_url) {
        return;
      }
      item.showOriginal = !item.showOriginal;
      renderCurrentQuestion();
    }

    function renderBookmarksPayload(items) {
      if (!items || !items.length) {
        return renderEmpty('No bookmarks yet for this subject.');
      }
      return items.map((item, index) => `
        <button class="topic-card" data-bookmark-index="${escapeHtml(index)}" type="button">
          <div class="result-meta">
            <span>${escapeHtml(item.document || '-')}</span>
            <span>page ${escapeHtml(item.page ?? '-')}</span>
            ${item.question_number ? `<span class="chip">${escapeHtml(item.question_number)}</span>` : ''}
          </div>
          <div class="result-text">${escapeHtml((item.text || '').slice(0, 180))}</div>
          ${item.primary_topic ? `<div class="panel-note">${escapeHtml(item.primary_topic)}</div>` : ''}
        </button>
      `).join('');
    }

    function updateQuestionNavigation() {
      const total = currentTopicQuestions.length;
      if (!total) {
        questionCounter.textContent = '0 questions';
        prevQuestionBtn.disabled = true;
        nextQuestionBtn.disabled = true;
        return;
      }
      const current = currentQuestionIndex + 1;
      const remaining = total - current;
      questionCounter.textContent = `${current} of ${total} • ${remaining} more after this`;
      prevQuestionBtn.disabled = currentQuestionIndex === 0;
      nextQuestionBtn.disabled = currentQuestionIndex >= total - 1;
    }

    function syncSidebar() {
      topicSidebar.classList.toggle('open', sidebarOpen);
      sidebarBackdrop.classList.toggle('visible', sidebarOpen);
      topicSidebar.setAttribute('aria-hidden', sidebarOpen ? 'false' : 'true');
    }

    function setSidebarOpen(value) {
      sidebarOpen = Boolean(value);
      syncSidebarMode();
      syncSidebar();
    }

    function syncTheme() {
      const isDark = activeTheme === 'dark';
      document.body.classList.toggle('dark-mode', isDark);
      themeToggleBtn.textContent = isDark ? '☀' : '☾';
      themeToggleBtn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
      themeToggleBtn.setAttribute('title', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    }

    function adminHeaders(extra = { }) {
      return adminKey ? { ...extra, 'x-admin-key': adminKey } : extra;
    }

    async function ensureAdminAccess() {
      if (!adminProtected) {
        return true;
      }
      if (adminKey) {
        return true;
      }
      const entered = window.prompt('Enter admin key');
      if (!entered) {
        return false;
      }
      adminKey = entered.trim();
      if (!adminKey) {
        return false;
      }
      sessionStorage.setItem('study.adminKey', adminKey);
      return true;
    }

    function syncMode() {
      const isAdmin = activeMode === 'admin';
      adminView.classList.toggle('hidden', !isAdmin);
      practiceView.classList.toggle('hidden', isAdmin);
      practiceModeBtn.classList.toggle('active', !isAdmin);
      adminModeBtn.classList.toggle('active', isAdmin);
      if (isAdmin) {
        setSidebarOpen(false);
        subjectHint.textContent = currentSubject
          ? `Admin mode for ${currentSubject}. Workflow: save syllabus, upload cleaned JSON, then upload original PDF.`
          : 'Admin mode. Open a subject to manage its content.';
        return;
      }
      subjectHint.textContent = currentSubject
        ? `Practice mode for ${currentSubject}. Open a topic and work through the questions.`
        : 'Practice mode. Open a subject to start.';
    }

    function renderCurrentQuestion() {
      topicQuestions.innerHTML = renderQuestionViewer(currentTopicQuestions, currentQuestionIndex);
      updateQuestionNavigation();
    }

    function renderIngestPayload(payload) {
      const indexed = payload.indexed || [];
      const errors = payload.errors || [];
      const cards = indexed.map(item => {
        const warning = item.source_type === 'pyq' && item.questions === 0
          ? '<div class="result-text">No solid question blocks were extracted. Keep the cheap vision fallback enabled for scanned papers.</div>'
          : '';
        return `
          <article class="result-card">
            <div class="result-meta">
              <span class="chip">${escapeHtml(item.subject || 'general')}</span>
              <span>${escapeHtml(item.source_type || '-')}</span>
              <span>${escapeHtml(item.document || '-')}</span>
            </div>
            <div class="result-text">Indexed ${escapeHtml(String(item.pages ?? 0))} pages, ${escapeHtml(String(item.chunks ?? 0))} chunks, ${escapeHtml(String(item.questions ?? 0))} questions.</div>
            ${warning}
            <div class="result-path">${escapeHtml(item.saved_path || '')}</div>
          </article>
        `;
      });
      const errorCards = errors.map(item => `
        <article class="result-card">
          <div class="result-meta"><span class="chip">Error</span><span>${escapeHtml(item.file || '-')}</span></div>
          <div class="result-text">${escapeHtml(item.error || 'Unknown error')}</div>
        </article>
      `);
      return [...cards, ...errorCards].join('') || renderEmpty('No uploads yet.');
    }

    function renderTopicPayload(payload) {
      const topics = payload.question_topics || [];
      if (!topics.length) {
        return renderEmpty('No topics found for this subject yet.');
      }
      return topics.map(item => `
        <button class="topic-card ${item.topic === currentTopic ? 'active' : ''}" data-topic="${escapeHtml(item.topic)}" type="button">
          <div class="result-meta">
            <span>frequency ${escapeHtml(String(item.question_count ?? item.count ?? 0))}</span>
            <span>${escapeHtml(String(item.chunk_count ?? 0))} supporting chunks</span>
          </div>
          <div class="result-text">${escapeHtml(item.topic)}</div>
          ${(item.examples || []).length ? `<div class="panel-note">${escapeHtml(item.examples[0])}</div>` : ''}
        </button>
      `).join('');
    }

    function renderSubjectTabs() {
      if (!knownSubjects.length) {
        subjectTabs.innerHTML = renderEmpty('No subjects yet.');
        return;
      }
      subjectTabs.innerHTML = knownSubjects.map(item => `
        <button class="tab ${item.subject === currentSubject ? 'active' : ''}" data-subject="${escapeHtml(item.subject)}">
          ${escapeHtml(item.subject)} (${escapeHtml(String(item.document_count))})
        </button>
      `).join('');
      subjectTabs.querySelectorAll('[data-subject]').forEach(button => {
        button.addEventListener('click', () => setCurrentSubject(button.dataset.subject));
      });
    }

    function renderFiles() {
      fileList.innerHTML = '';
      selectedFiles.forEach(file => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
        fileList.appendChild(item);
      });
      if (!selectedFiles.length) {
        fileList.innerHTML = '<div class="file-item">No files selected.</div>';
      }
    }

    function setStatus(value) {
      if (typeof value === 'string') {
        statusBox.textContent = value;
        return;
      }
      statusBox.innerHTML = renderIngestPayload(value);
    }

    function setCurrentSubject(subject) {
      currentSubject = (subject || '').trim();
      if (!currentSubject) return;
      currentTopic = '';
      localStorage.setItem('study.currentSubject', currentSubject);
      subjectInput.value = currentSubject;
      currentSubjectLabel.textContent = currentSubject;
      emptyState.classList.add('hidden');
      subjectDashboard.classList.remove('hidden');
      setSidebarOpen(false);
      syncMode();
      renderSubjectTabs();
      loadSubjectOverview();
      loadStats();
    }

    function openSubjectFromInput() {
      const value = subjectInput.value.trim();
      if (!value) {
        subjectHint.textContent = 'Enter a subject name first.';
        return;
      }
      if (!knownSubjects.find(item => item.subject === value)) {
        knownSubjects = [{ subject: value, document_count: 0 }, ...knownSubjects];
      }
      setCurrentSubject(value);
    }

    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
      selectedFiles = Array.from(fileInput.files);
      renderFiles();
    });
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, event => {
        event.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', event => {
      selectedFiles = Array.from(event.dataTransfer.files);
      renderFiles();
    });

    async function uploadFiles() {
      if (!(await ensureAdminAccess())) {
        return;
      }
      if (!currentSubject) {
        setStatus('Create or select a subject first.');
        return;
      }
      if (!syllabusInput.value.trim()) {
        setStatus('Step 1 is required: paste and save the syllabus before uploading files.');
        return;
      }
      if (!selectedFiles.length) {
        setStatus('Choose at least one file.');
        return;
      }
      const hasJson = selectedFiles.some(file => file.name.toLowerCase().endsWith('.json'));
      const hasPdf = selectedFiles.some(file => file.name.toLowerCase().endsWith('.pdf'));
      if (hasJson && hasPdf) {
        setStatus('Upload the cleaned JSON first and the original PDF in a separate upload.');
        return;
      }
      if (hasPdf && Number(latestSubjectOverview?.totals?.questions ?? 0) === 0) {
        setStatus('Upload the cleaned JSON first. The original PDF should come after the questions exist for this subject.');
        return;
      }

      const formData = new FormData();
      formData.append('subject', currentSubject);
      formData.append('source_type', document.getElementById('sourceType').value);
      formData.append('vision_provider', document.getElementById('visionProvider').value);
      formData.append('ocr_lang', document.getElementById('ocrLang').value);
      selectedFiles.forEach(file => formData.append('files', file));

      setStatus(hasJson ? 'Uploading cleaned JSON...' : 'Uploading original PDF...');
      const response = await fetch('/api/ingest', { method: 'POST', body: formData, headers: adminHeaders() });
      const payload = await response.json();
      if (response.status === 403) {
        adminKey = '';
        sessionStorage.removeItem('study.adminKey');
      }
      setStatus(payload);
      if (response.ok) {
        await loadStats();
        await loadSubjectOverview();
      }
    }

    async function loadTopicQuestions(topic) {
      if (!currentSubject || !topic) {
        topicQuestions.innerHTML = renderEmpty('Choose a topic from the left.');
        return;
      }
      currentTopic = topic;
      sidebarMode = 'topics';
      showingBookmarks = false;
      currentTopicQuestions = [];
      currentQuestionIndex = 0;
      selectedTopicLabel.textContent = topic;
      questionHint.textContent = `Showing extracted PYQ questions tagged under ${topic}.`;
      topicResults.innerHTML = renderTopicPayload(latestSubjectOverview || { question_topics: [] });
      topicResults.querySelectorAll('[data-topic]').forEach(button => {
        button.addEventListener('click', () => loadTopicQuestions(button.dataset.topic));
      });
      syncSidebarMode();
      topicQuestions.innerHTML = renderEmpty('Loading questions...');
      updateQuestionNavigation();
      const params = new URLSearchParams({ subject: currentSubject, topic, limit: '120' });
      const response = await fetch(`/api/topic-questions?${params.toString()}`);
      const payload = await response.json();
      currentTopicQuestions = payload.results || [];
      currentQuestionIndex = 0;
      renderCurrentQuestion();
      setSidebarOpen(false);
    }

    async function loadSubjectOverview() {
      if (!currentSubject) {
        return;
      }
      const response = await fetch(`/api/subject-overview?subject=${encodeURIComponent(currentSubject)}`);
      const payload = await response.json();
      latestSubjectOverview = payload;
      sidebarMode = 'topics';
      topicResults.innerHTML = renderTopicPayload(payload);
      topicResults.querySelectorAll('[data-topic]').forEach(button => {
        button.addEventListener('click', () => loadTopicQuestions(button.dataset.topic));
      });
      overviewQuestions.textContent = String(payload.totals?.questions ?? 0);
      overviewDocuments.textContent = String(payload.totals?.documents ?? 0);
      syllabusInput.value = payload.syllabus_text || '';
      const topics = payload.question_topics || [];
      if (topics.length) {
        const nextTopic = topics.find(item => item.topic === currentTopic)?.topic || topics[0].topic;
        await loadTopicQuestions(nextTopic);
      } else {
        currentTopic = '';
        currentTopicQuestions = [];
        currentQuestionIndex = 0;
        selectedTopicLabel.textContent = 'Questions';
        questionHint.textContent = 'Choose a topic from the left.';
        topicQuestions.innerHTML = renderEmpty('No questions yet.');
        updateQuestionNavigation();
      }
    }

    async function loadStats() {
      const response = await fetch('/api/stats');
      const payload = await response.json();
      knownSubjects = payload.subjects || [];
      renderSubjectTabs();
    }

    async function saveSyllabus() {
      if (!(await ensureAdminAccess())) {
        return;
      }
      if (!currentSubject) {
        setStatus('Open a subject first.');
        return;
      }
      const response = await fetch('/api/syllabus', {
        method: 'POST',
        headers: adminHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          subject: currentSubject,
          syllabus_text: syllabusInput.value || '',
        }),
      });
      const payload = await response.json();
      if (response.status === 403) {
        adminKey = '';
        sessionStorage.removeItem('study.adminKey');
      }
      if (!response.ok) {
        setStatus(payload.detail || 'Could not save syllabus.');
        return;
      }
      setStatus('Syllabus saved. Future topic mapping will stay within these syllabus topics.');
      await loadSubjectOverview();
    }

    async function loadBookmarks() {
      if (!currentSubject) {
        return;
      }
      sidebarMode = 'bookmarks';
      showingBookmarks = true;
      currentTopic = '';
      const response = await fetch(`/api/bookmarks?subject=${encodeURIComponent(currentSubject)}`);
      const payload = await response.json();
      currentTopicQuestions = payload.results || [];
      topicResults.innerHTML = renderBookmarksPayload(currentTopicQuestions);
      topicResults.querySelectorAll('[data-bookmark-index]').forEach(button => {
        button.addEventListener('click', () => {
          const nextIndex = Number(button.dataset.bookmarkIndex || '0');
          currentQuestionIndex = Number.isFinite(nextIndex) ? nextIndex : 0;
          selectedTopicLabel.textContent = 'Bookmarked Questions';
          questionHint.textContent = `Saved questions for ${currentSubject}.`;
          renderCurrentQuestion();
          setSidebarOpen(false);
        });
      });
      syncSidebarMode();
      if (!currentTopicQuestions.length) {
        selectedTopicLabel.textContent = 'Bookmarked Questions';
        questionHint.textContent = `Saved questions for ${currentSubject}.`;
        topicQuestions.innerHTML = renderEmpty('No bookmarks yet.');
        updateQuestionNavigation();
        return;
      }
      currentQuestionIndex = 0;
      selectedTopicLabel.textContent = 'Bookmarked Questions';
      questionHint.textContent = `Saved questions for ${currentSubject}.`;
      renderCurrentQuestion();
    }

    async function deleteSubject() {
      if (!(await ensureAdminAccess())) {
        return;
      }
      if (!currentSubject) {
        setStatus('Select a subject first.');
        return;
      }
      if (!window.confirm(`Delete subject "${currentSubject}" and all its uploaded files?`)) {
        return;
      }
      const response = await fetch(`/api/subjects/${encodeURIComponent(currentSubject)}`, {
        method: 'DELETE',
        headers: adminHeaders(),
      });
      const payload = await response.json();
      if (response.status === 403) {
        adminKey = '';
        sessionStorage.removeItem('study.adminKey');
      }
      setStatus(typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2));
      currentSubject = '';
      localStorage.removeItem('study.currentSubject');
      currentSubjectLabel.textContent = '-';
      subjectInput.value = '';
      emptyState.classList.remove('hidden');
      subjectDashboard.classList.add('hidden');
      setSidebarOpen(false);
      currentTopicQuestions = [];
      currentQuestionIndex = 0;
      showingBookmarks = false;
      sidebarMode = 'topics';
      topicResults.innerHTML = renderEmpty('No topics found for this subject yet.');
      topicQuestions.innerHTML = renderEmpty('No questions yet.');
      selectedTopicLabel.textContent = 'Questions';
      questionHint.textContent = 'Choose a topic from the left.';
      updateQuestionNavigation();
      syncMode();
      await loadStats();
    }

    async function resetDatabase() {
      if (!(await ensureAdminAccess())) {
        return;
      }
      if (!window.confirm('Delete the entire study database and all uploaded files?')) {
        return;
      }
      const response = await fetch('/api/database/reset', { method: 'POST', headers: adminHeaders() });
      const payload = await response.json();
      if (response.status === 403) {
        adminKey = '';
        sessionStorage.removeItem('study.adminKey');
      }
      setStatus(typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2));
      currentSubject = '';
      knownSubjects = [];
      localStorage.removeItem('study.currentSubject');
      currentSubjectLabel.textContent = '-';
      subjectInput.value = '';
      subjectHint.textContent = 'Create or select a subject tab to begin.';
      emptyState.classList.remove('hidden');
      subjectDashboard.classList.add('hidden');
      setSidebarOpen(false);
      currentTopicQuestions = [];
      currentQuestionIndex = 0;
      showingBookmarks = false;
      sidebarMode = 'topics';
      topicResults.innerHTML = renderEmpty('No topics found for this subject yet.');
      topicQuestions.innerHTML = renderEmpty('No questions yet.');
      selectedTopicLabel.textContent = 'Questions';
      questionHint.textContent = 'Choose a topic from the left.';
      updateQuestionNavigation();
      renderSubjectTabs();
      syncMode();
      await loadStats();
    }

    document.getElementById('saveSubjectBtn').addEventListener('click', openSubjectFromInput);
    document.getElementById('uploadBtn').addEventListener('click', uploadFiles);
    document.getElementById('saveSyllabusBtn').addEventListener('click', saveSyllabus);
    document.getElementById('refreshSubjectBtn').addEventListener('click', loadSubjectOverview);
    document.getElementById('deleteSubjectBtn').addEventListener('click', deleteSubject);
    document.getElementById('resetDbBtn').addEventListener('click', resetDatabase);
    practiceModeBtn.addEventListener('click', () => {
      activeMode = 'practice';
      localStorage.setItem('study.activeMode', activeMode);
      syncMode();
    });
    adminModeBtn.addEventListener('click', () => {
      ensureAdminAccess().then(ok => {
        if (!ok) {
          return;
        }
        activeMode = 'admin';
        localStorage.setItem('study.activeMode', activeMode);
        syncMode();
      });
    });
    themeToggleBtn.addEventListener('click', () => {
      activeTheme = activeTheme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('study.theme', activeTheme);
      syncTheme();
    });
    openSidebarBtn.addEventListener('click', () => setSidebarOpen(true));
    closeSidebarBtn.addEventListener('click', () => setSidebarOpen(false));
    sidebarBackdrop.addEventListener('click', () => setSidebarOpen(false));
    prevQuestionBtn.addEventListener('click', () => {
      if (currentQuestionIndex > 0) {
        currentQuestionIndex -= 1;
        renderCurrentQuestion();
      }
    });
    nextQuestionBtn.addEventListener('click', () => {
      if (currentQuestionIndex < currentTopicQuestions.length - 1) {
        currentQuestionIndex += 1;
        renderCurrentQuestion();
      }
    });
    sidebarTopicsBtn.addEventListener('click', async () => {
      sidebarMode = 'topics';
      syncSidebarMode();
      if (latestSubjectOverview) {
        topicResults.innerHTML = renderTopicPayload(latestSubjectOverview);
        topicResults.querySelectorAll('[data-topic]').forEach(button => {
          button.addEventListener('click', () => loadTopicQuestions(button.dataset.topic));
        });
      }
    });
    sidebarBookmarksBtn.addEventListener('click', loadBookmarks);
    topicQuestions.addEventListener('click', event => {
      const originalButton = event.target.closest('[data-toggle-original]');
      if (originalButton) {
        toggleOriginalCurrentQuestion();
        return;
      }
      const bookmarkButton = event.target.closest('[data-bookmark-question]');
      if (bookmarkButton) {
        toggleBookmarkCurrentQuestion(bookmarkButton);
        return;
      }
      const button = event.target.closest('[data-copy-question]');
      if (!button) {
        return;
      }
      copyCurrentQuestion(button);
    });
    subjectInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') openSubjectFromInput();
      if (event.key === 'Escape') setSidebarOpen(false);
    });

    renderFiles();
    syncSidebar();
    syncTheme();
    syncMode();
    updateQuestionNavigation();
    loadStats();
    if (currentSubject) {
      setCurrentSubject(currentSubject);
    }
  </script>
</body>
</html>
    """.replace("__ADMIN_PROTECTED__", admin_protected)


@app.post("/api/ingest")
async def api_ingest(
    subject: str = Form(...),
    source_type: str = Form(...),
    vision_provider: str = Form("none"),
    ocr_lang: str = Form("eng"),
    files: list[UploadFile] = File(...),
    x_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    verify_admin(x_admin_key)
    if source_type not in {"pyq", "reference", "handout", "notes", "ppt"}:
        raise HTTPException(status_code=400, detail="Invalid source_type")
    if vision_provider not in {"none", "claude_cheap", "claude"}:
        raise HTTPException(status_code=400, detail="Invalid vision_provider")

    results = []
    errors = []
    for file in files:
        try:
            saved_path = save_upload(subject, source_type, file)
            results.append(
                ingest_path(
                    saved_path,
                    subject=subject,
                    source_type=source_type,
                    vision_provider=vision_provider,
                    ocr_lang=ocr_lang,
                    chunk_size=1000,
                )
            )
        except Exception as exc:
            errors.append({"file": file.filename, "error": str(exc)})
        finally:
            await file.close()

    status_code = 200 if not errors else 207
    return JSONResponse({"indexed": results, "errors": errors}, status_code=status_code)


@app.get("/api/search")
def api_search(query: str, subject: str | None = None, limit: int = 8) -> dict:
    store = get_store()
    try:
        results = store.search(query, limit=limit, subject=subject)
        return {
            "query": query,
            "subject": subject,
            "results": [
                {
                    "subject": item.subject,
                    "document": item.document_name,
                    "page": item.page_number,
                    "score": round(item.score, 4),
                    "topics": canonicalize_topics(item.topics),
                    "path": item.path,
                    "text": item.chunk_text,
                }
                for item in results
            ],
        }
    finally:
        store.close()


@app.get("/api/questions")
def api_questions(query: str, subject: str | None = None, limit: int = 10) -> dict:
    store = get_store()
    try:
        rows = store.question_search(query, limit=limit, subject=subject)
        return {
            "query": query,
            "subject": subject,
            "results": [question_payload(row) for row in rows],
        }
    finally:
        store.close()


@app.get("/api/subject-overview")
def api_subject_overview(subject: str, question_limit: int = 120, topic_limit: int = 50) -> dict:
    store = get_store()
    try:
        overview = store.subject_overview(
            subject,
            question_limit=question_limit,
            topic_limit=topic_limit,
            include_questions=False,
        )
        return {
            "subject": subject,
            "totals": {
                "documents": overview["documents"],
                "chunks": overview["chunks"],
                "questions": overview["questions"],
            },
            "question_topics": [
                topic
                for topic in overview["question_topics"]
            ],
            "topic_catalog": store.get_subject_topic_catalog(subject),
            "syllabus_text": store.get_subject_syllabus(subject),
            "questions": [],
        }
    finally:
        store.close()


@app.get("/api/topic-questions")
def api_topic_questions(subject: str, topic: str, limit: int = 120) -> dict:
    store = get_store()
    try:
        rows = store.questions_for_topic(subject, topic, limit=limit)
        return {
            "subject": subject,
            "topic": topic,
            "results": [
                {
                    **question_payload(row),
                    "bookmarked": store.bookmark_exists(
                        subject=row["subject"],
                        document_path=row["path"],
                        page_number=row["page_number"],
                        question_number=row["question_number"],
                        text=row["text"],
                    ),
                }
                for row in rows
            ],
        }
    finally:
        store.close()


@app.get("/api/bookmarks")
def api_bookmarks(subject: str, limit: int = 200) -> dict:
    store = get_store()
    try:
        rows = store.bookmarks(subject, limit=limit)
        return {
            "subject": subject,
            "results": [
                {
                    "subject": row["subject"],
                    "source_type": row["source_type"],
                    "document": row["document_name"],
                    "page": row["page_number"],
                    "question_number": row["question_number"],
                    "primary_topic": row["primary_topic"],
                    "has_diagram": bool(row["has_diagram"]),
                    "topics": canonicalize_topics(json.loads(row["topics_json"])),
                    "path": row["document_path"],
                    "text": row["text"],
                    "source_pdf_path": row["source_pdf_path"],
                    "source_pdf_name": row["source_pdf_name"],
                    "source_pdf_page": row["source_pdf_page"],
                    "original_image_path": row["original_image_path"],
                    "diagram_image_path": row["diagram_image_path"],
                    "original_view_url": asset_url(row["original_image_path"]),
                    "diagram_url": asset_url(row["diagram_image_path"]),
                    "link_confidence": row["link_confidence"],
                    "bookmarked": True,
                }
                for row in rows
            ],
        }
    finally:
        store.close()


@app.post("/api/bookmarks")
def api_add_bookmark(payload: dict) -> dict:
    store = get_store()
    try:
        store.add_bookmark(
            subject=str(payload.get("subject") or "").strip(),
            document_path=str(payload.get("path") or ""),
            document_name=str(payload.get("document") or ""),
            source_type=str(payload.get("source_type") or "") or None,
            page_number=int(payload.get("page") or 0),
            question_number=str(payload.get("question_number")) if payload.get("question_number") is not None else None,
            text=str(payload.get("text") or ""),
            primary_topic=str(payload.get("primary_topic") or "") or None,
            has_diagram=bool(payload.get("has_diagram")),
            source_pdf_path=str(payload.get("source_pdf_path") or "") or None,
            source_pdf_name=str(payload.get("source_pdf_name") or "") or None,
            source_pdf_page=int(payload.get("source_pdf_page")) if payload.get("source_pdf_page") is not None else None,
            original_image_path=str(payload.get("original_image_path") or "") or None,
            diagram_image_path=str(payload.get("diagram_image_path") or "") or None,
            link_confidence=float(payload.get("link_confidence")) if payload.get("link_confidence") is not None else None,
            topics=[str(item) for item in (payload.get("topics") or [])],
        )
        return {"status": "bookmarked"}
    finally:
        store.close()


@app.delete("/api/bookmarks")
def api_remove_bookmark(payload: dict) -> dict:
    store = get_store()
    try:
        store.remove_bookmark(
            subject=str(payload.get("subject") or "").strip(),
            document_path=str(payload.get("path") or ""),
            page_number=int(payload.get("page") or 0),
            question_number=str(payload.get("question_number")) if payload.get("question_number") is not None else None,
            text=str(payload.get("text") or ""),
        )
        return {"status": "removed"}
    finally:
        store.close()


@app.get("/api/topics")
def api_topics(subject: str, limit: int = 15) -> dict:
    store = get_store()
    try:
        return {
            "subject": subject,
            "topics": [
                {"topic": topic, "count": count}
                for topic, count in store.subject_topics(subject, limit=limit)
            ],
            "catalog": store.get_subject_topic_catalog(subject),
        }
    finally:
        store.close()


@app.post("/api/syllabus")
def api_save_syllabus(payload: dict, x_admin_key: str | None = Header(default=None)) -> dict:
    verify_admin(x_admin_key)
    subject = str(payload.get("subject") or "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    syllabus_text = str(payload.get("syllabus_text") or "")
    from study_pipeline.topic_classifier import syllabus_topics_from_text

    store = get_store()
    try:
        store.set_subject_syllabus(subject, syllabus_text)
        syllabus_topics = syllabus_topics_from_text(syllabus_text)
        if syllabus_topics:
            store.set_subject_topic_catalog(subject, syllabus_topics)
        return {
            "subject": subject,
            "syllabus_text": store.get_subject_syllabus(subject),
            "topic_catalog": store.get_subject_topic_catalog(subject),
            "status": "saved",
        }
    finally:
        store.close()


@app.get("/api/stats")
def api_stats() -> dict:
    store = get_store()
    try:
        return {
            "totals": store.stats(),
            "subjects": [dict(row) for row in store.subject_stats()],
        }
    finally:
        store.close()


@app.get("/api/assets/original")
def api_original_asset(path: str) -> FileResponse:
    if not path.strip():
        raise HTTPException(status_code=400, detail="path is required")
    asset_path = (ORIGINAL_ASSET_ROOT / path).resolve()
    try:
        asset_path.relative_to(ORIGINAL_ASSET_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid asset path") from exc
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(asset_path)


@app.delete("/api/subjects/{subject}")
def api_delete_subject(subject: str, x_admin_key: str | None = Header(default=None)) -> dict:
    verify_admin(x_admin_key)
    subject = subject.strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    store = get_store()
    try:
        deleted = store.delete_subject(subject)
    finally:
        store.close()
    remove_subject_uploads(subject)
    return {
        "subject": subject,
        "deleted": deleted,
        "status": "subject_deleted",
        "db_path": str(DB_PATH),
    }


@app.post("/api/subjects/delete")
def api_delete_subject_compat(payload: dict, x_admin_key: str | None = Header(default=None)) -> dict:
    verify_admin(x_admin_key)
    subject = str(payload.get("subject") or "").strip()
    return api_delete_subject(subject, x_admin_key=x_admin_key)


@app.post("/api/database/reset")
def api_reset_database(x_admin_key: str | None = Header(default=None)) -> dict:
    verify_admin(x_admin_key)
    store = get_store()
    try:
        deleted = store.reset_all()
    finally:
        store.close()
    remove_database_file()
    remove_all_uploads()
    return {
        "deleted": deleted,
        "status": "database_reset",
        "db_path": str(DB_PATH),
    }
