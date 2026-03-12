from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from study_pipeline.embeddings import HashingEmbeddingProvider, OllamaEmbeddingProvider
from study_pipeline.store import StudyStore


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = BASE_DIR / "data" / "uploads"
DB_PATH = Path(os.environ.get("STUDY_DB_PATH", BASE_DIR / "data" / "processed" / "study_index.db"))
EMBEDDING_PROVIDER = os.environ.get("STUDY_EMBEDDING_PROVIDER", "hashing")
OLLAMA_MODEL = os.environ.get("STUDY_OLLAMA_MODEL", "nomic-embed-text")
OLLAMA_URL = os.environ.get("STUDY_OLLAMA_URL", "http://localhost:11434")

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


def ingest_path(path: Path, subject: str, source_type: str, vision_provider: str, ocr_lang: str, chunk_size: int) -> dict:
    from study_pipeline.chunking import chunk_page
    from study_pipeline.extract import extract_document
    from study_pipeline.json_input import load_question_json
    from study_pipeline.questions import extract_questions
    from study_pipeline.vision import extract_pyq_questions_with_vision_fallback

    if path.suffix.lower() == ".json":
        pages, questions = load_question_json(path)
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
    store.upsert_document(
        path=str(path),
        name=path.name,
        subject=subject,
        kind=path.suffix.lower().lstrip("."),
        source_type=source_type,
        chunks=chunks,
        questions=questions,
    )
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
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Study Index</title>
  <style>
    :root {
      --ink: #182033;
      --muted: #5d6470;
      --line: #d6cdbd;
      --panel: rgba(255, 252, 246, 0.94);
      --panel-strong: #fffdfa;
      --accent: #0f4c81;
      --accent-2: #0d9488;
      --accent-3: #aa5a11;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 76, 129, 0.12), transparent 26%),
        radial-gradient(circle at bottom right, rgba(170, 90, 17, 0.18), transparent 24%),
        linear-gradient(180deg, #f8f4ec 0%, #ede3d2 100%);
    }
    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }
    .hero {
      display: grid;
      gap: 16px;
      margin-bottom: 20px;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(2.4rem, 4vw, 4.8rem);
      line-height: 0.9;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 0;
      max-width: 900px;
      color: var(--muted);
      font-size: 1.04rem;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(31,41,55,0.06);
      backdrop-filter: blur(6px);
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 1.08rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .panel-note {
      color: var(--muted);
      font-size: 0.93rem;
      line-height: 1.45;
    }
    .subject-bar, .stack, .field, .overview-strip, .results, .status {
      display: grid;
      gap: 12px;
    }
    .subject-row, .toolbar, .tab-row, .topic-row, .result-meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .dashboard {
      display: grid;
      grid-template-columns: 1.15fr 0.95fr;
      gap: 18px;
    }
    .overview-strip {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin-bottom: 18px;
    }
    .hero-card, .stat-card {
      border: 1px solid rgba(15, 76, 129, 0.14);
      border-radius: 18px;
      padding: 16px;
      background: linear-gradient(140deg, rgba(255,255,255,0.92), rgba(238,246,244,0.84));
    }
    .stat-label {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    .stat-value {
      margin-top: 6px;
      font-size: 2rem;
      font-weight: 700;
    }
    label {
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    input, select, button { font: inherit; }
    input, select {
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
    }
    .subject-row input { max-width: 280px; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      color: white;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      cursor: pointer;
      font-weight: 700;
    }
    button.secondary {
      background: linear-gradient(90deg, #8f4c16, var(--accent-3));
    }
    .tab {
      border: 1px solid rgba(15, 76, 129, 0.18);
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(255,255,255,0.8);
      color: var(--ink);
      cursor: pointer;
      font-weight: 700;
    }
    .tab.active {
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: white;
      border-color: transparent;
    }
    .dropzone {
      border: 2px dashed rgba(15, 76, 129, 0.35);
      border-radius: 18px;
      padding: 26px;
      min-height: 180px;
      display: grid;
      place-items: center;
      text-align: center;
      background: linear-gradient(135deg, rgba(15, 76, 129, 0.08), rgba(13, 148, 136, 0.08));
      transition: 160ms ease;
    }
    .dropzone.dragover {
      transform: translateY(-2px);
      border-color: var(--accent-2);
      background: linear-gradient(135deg, rgba(15, 76, 129, 0.14), rgba(13, 148, 136, 0.14));
    }
    .dropzone strong {
      display: block;
      font-size: 1.15rem;
      margin-bottom: 6px;
    }
    .files {
      margin-top: 12px;
      display: grid;
      gap: 8px;
      font-size: 0.95rem;
    }
    .file-item {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
    }
    .results, .status {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      min-height: 140px;
      overflow: auto;
    }
    .status {
      white-space: pre-wrap;
      font-size: 0.95rem;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
    .result-card {
      border: 1px solid rgba(216, 208, 191, 0.9);
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,244,236,0.9));
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .result-meta {
      font-size: 0.82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .chip, .topic-pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-weight: 700;
    }
    .chip {
      background: rgba(15, 76, 129, 0.08);
      color: var(--accent);
    }
    .topic-pill {
      background: rgba(13, 148, 136, 0.10);
      color: var(--accent-2);
      font-size: 0.82rem;
    }
    .score {
      color: var(--accent-2);
      font-weight: 700;
    }
    .result-text {
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 1rem;
    }
    .result-path {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.8rem;
      color: var(--muted);
      word-break: break-all;
    }
    .stats-grid, .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .hidden { display: none; }
    @media (max-width: 920px) {
      .dashboard, .overview-strip, .stats-grid, .mini-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>One subject. Every asked question.</h1>
      <p>Start by naming a subject once. Then upload the best PDF containing all questions, extract clean question blocks, search inside the vector store, and see how many times each topic has been asked so far.</p>
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

    <section id="emptyState" class="panel stack">
      <h2>Start Here</h2>
      <div class="panel-note">The first thing this app asks for is the subject name. After that, everything stays grouped inside subject tabs. Upload your best question PDF to build one clean subject dashboard.</div>
    </section>

    <section id="subjectDashboard" class="hidden">
      <div class="overview-strip">
        <div class="hero-card">
          <div class="stat-label">Current Subject</div>
          <div id="currentSubjectLabel" class="stat-value">-</div>
        </div>
        <div class="hero-card">
          <div class="stat-label">Questions Extracted</div>
          <div id="overviewQuestions" class="stat-value">0</div>
        </div>
        <div class="hero-card">
          <div class="stat-label">Indexed Documents</div>
          <div id="overviewDocuments" class="stat-value">0</div>
        </div>
      </div>
      <section class="panel stack">
        <h2>Danger Zone</h2>
        <div class="panel-note">Delete only the current subject or wipe the entire study database and uploaded files.</div>
        <div class="toolbar">
          <button id="deleteSubjectBtn" class="secondary">Delete This Subject</button>
          <button id="resetDbBtn" class="secondary">Delete Entire Database</button>
        </div>
      </section>

      <section class="dashboard">
        <div class="stack">
          <div class="panel stack">
            <h2>Upload Question PDF</h2>
            <div class="panel-note">Upload your main PYQ PDF here. The app uses text extraction first and falls back to the cheap vision path only when no usable question blocks are found.</div>
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
                  <option value="claude_cheap">Claude Cheap Fallback</option>
                  <option value="none">None</option>
                  <option value="claude">Claude</option>
                </select>
              </div>
              <div class="field">
                <label for="ocrLang">OCR Language</label>
                <input id="ocrLang" value="eng">
              </div>
            </div>
            <div id="dropzone" class="dropzone">
              <div>
                <strong>Drop PDFs, JSONs, PPTXs, MDs, or TXTs here</strong>
                <div>Use a clean question JSON if you already have one.</div>
                <input id="fileInput" type="file" multiple hidden accept=".pdf,.json,.pptx,.txt,.md">
                <div id="fileList" class="files"></div>
              </div>
            </div>
            <div class="toolbar">
              <button id="uploadBtn">Ingest Files</button>
              <button id="refreshSubjectBtn" class="secondary">Refresh Subject</button>
            </div>
            <div class="field">
              <label>Status</label>
              <div id="status" class="status">No uploads yet.</div>
            </div>
          </div>

          <div class="panel stack">
            <h2>All Extracted Questions</h2>
            <div class="panel-note">Every stored question for the current subject, shown cleanly instead of raw JSON.</div>
            <div id="allQuestions" class="results">No questions yet.</div>
          </div>
        </div>

        <div class="stack">
          <div class="panel stack">
            <h2>Topic Frequency</h2>
            <div class="panel-note">This uses stored extracted questions to count how many questions from each topic were asked till now.</div>
            <div id="topicResults" class="results">No topics yet.</div>
          </div>

          <div class="panel stack">
            <h2>Search Within Subject</h2>
            <div class="field">
              <label for="searchQuery">Topic Query</label>
              <input id="searchQuery" placeholder="normalization, cache coherence, pipelining">
            </div>
            <div class="toolbar">
              <button id="searchBtn">Search Material</button>
              <button id="questionBtn" class="secondary">Search Questions</button>
            </div>
            <div id="searchResults" class="results">No results yet.</div>
          </div>

          <div class="panel stack">
            <h2>Database Stats</h2>
            <div id="statsResults" class="results">No stats loaded.</div>
          </div>
        </div>
      </section>
    </section>
  </div>

  <script>
    const emptyState = document.getElementById('emptyState');
    const subjectDashboard = document.getElementById('subjectDashboard');
    const subjectInput = document.getElementById('subjectInput');
    const subjectTabs = document.getElementById('subjectTabs');
    const subjectHint = document.getElementById('subjectHint');
    const currentSubjectLabel = document.getElementById('currentSubjectLabel');
    const overviewQuestions = document.getElementById('overviewQuestions');
    const overviewDocuments = document.getElementById('overviewDocuments');
    const allQuestions = document.getElementById('allQuestions');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const statusBox = document.getElementById('status');
    const searchResults = document.getElementById('searchResults');
    const topicResults = document.getElementById('topicResults');
    const statsResults = document.getElementById('statsResults');
    let selectedFiles = [];
    let knownSubjects = [];
    let currentSubject = localStorage.getItem('study.currentSubject') || '';

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

    function renderSearchPayload(payload, mode) {
      const items = payload.results || [];
      if (!items.length) {
        return renderEmpty(mode === 'questions' ? 'No matching questions found.' : 'No matching material found.');
      }
      return items.map(item => {
        const score = typeof item.score === 'number' ? `<span class="score">score ${item.score.toFixed(3)}</span>` : '';
        const number = item.question_number ? `<span class="chip">${escapeHtml(item.question_number)}</span>` : '';
        return `
          <article class="result-card">
            <div class="result-meta">
              <span class="chip">${escapeHtml(item.subject || 'general')}</span>
              <span>${escapeHtml(item.source_type || '')}</span>
              <span>${escapeHtml(item.document || 'Untitled')}</span>
              <span>page ${escapeHtml(item.page ?? '-')}</span>
              ${number}
              ${score}
            </div>
            <div class="result-text">${escapeHtml(item.text || '')}</div>
            ${renderTopics(item.topics || [])}
            <div class="result-path">${escapeHtml(item.path || '')}</div>
          </article>
        `;
      }).join('');
    }

    function renderQuestionList(items) {
      if (!items || !items.length) {
        return renderEmpty('No extracted questions for this subject yet.');
      }
      return items.map(item => `
        <article class="result-card">
          <div class="result-meta">
            <span class="chip">${escapeHtml(item.source_type || 'pyq')}</span>
            <span>${escapeHtml(item.document || '-')}</span>
            <span>page ${escapeHtml(item.page ?? '-')}</span>
            ${item.question_number ? `<span class="chip">${escapeHtml(item.question_number)}</span>` : ''}
          </div>
          <div class="result-text">${escapeHtml(item.text || '')}</div>
          ${renderTopics(item.topics || [])}
        </article>
      `).join('');
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
        <article class="result-card">
          <div class="result-meta">
            <span class="chip">${escapeHtml(payload.subject || 'general')}</span>
            <span>${escapeHtml(String(item.question_count ?? item.count ?? 0))} questions</span>
            <span>${escapeHtml(String(item.chunk_count ?? 0))} chunks</span>
            <span class="score">score ${escapeHtml(String(item.score ?? item.count ?? 0))}</span>
          </div>
          <div class="result-text">${escapeHtml(item.topic)}</div>
          ${(item.examples || []).length ? `<div class="panel-note">${item.examples.map(example => escapeHtml(example)).join(' | ')}</div>` : ''}
        </article>
      `).join('');
    }

    function renderStatsPayload(payload) {
      const totals = payload.totals || {};
      const subjects = payload.subjects || [];
      return `
        <div class="stats-grid">
          <div class="stat-card"><div class="stat-label">Documents</div><div class="stat-value">${escapeHtml(totals.documents ?? 0)}</div></div>
          <div class="stat-card"><div class="stat-label">Chunks</div><div class="stat-value">${escapeHtml(totals.chunks ?? 0)}</div></div>
          <div class="stat-card"><div class="stat-label">Questions</div><div class="stat-value">${escapeHtml(totals.questions ?? 0)}</div></div>
        </div>
        ${subjects.length ? subjects.map(item => `
          <article class="result-card">
            <div class="result-meta"><span class="chip">${escapeHtml(item.subject)}</span></div>
            <div class="result-text">${escapeHtml(String(item.document_count))} documents indexed</div>
          </article>
        `).join('') : renderEmpty('No subjects indexed yet.')}
      `;
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
      localStorage.setItem('study.currentSubject', currentSubject);
      subjectInput.value = currentSubject;
      currentSubjectLabel.textContent = currentSubject;
      subjectHint.textContent = `Working inside ${currentSubject}. Upload files and review extracted questions below.`;
      emptyState.classList.add('hidden');
      subjectDashboard.classList.remove('hidden');
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
      if (!currentSubject) {
        setStatus('Create or select a subject first.');
        return;
      }
      if (!selectedFiles.length) {
        setStatus('Choose at least one file.');
        return;
      }

      const formData = new FormData();
      formData.append('subject', currentSubject);
      formData.append('source_type', document.getElementById('sourceType').value);
      formData.append('vision_provider', document.getElementById('visionProvider').value);
      formData.append('ocr_lang', document.getElementById('ocrLang').value);
      selectedFiles.forEach(file => formData.append('files', file));

      setStatus('Uploading and indexing...');
      const response = await fetch('/api/ingest', { method: 'POST', body: formData });
      const payload = await response.json();
      setStatus(payload);
      if (response.ok) {
        await loadStats();
        await loadSubjectOverview();
      }
    }

    async function runSearch(mode) {
      const query = document.getElementById('searchQuery').value.trim();
      if (!currentSubject) {
        searchResults.innerHTML = renderEmpty('Select a subject first.');
        return;
      }
      if (!query) {
        searchResults.innerHTML = renderEmpty('Enter a query.');
        return;
      }
      const endpoint = mode === 'questions' ? '/api/questions' : '/api/search';
      const params = new URLSearchParams({ query, subject: currentSubject });
      const response = await fetch(`${endpoint}?${params.toString()}`);
      const payload = await response.json();
      searchResults.innerHTML = renderSearchPayload(payload, mode);
    }

    async function loadSubjectOverview() {
      if (!currentSubject) {
        return;
      }
      const response = await fetch(`/api/subject-overview?subject=${encodeURIComponent(currentSubject)}`);
      const payload = await response.json();
      topicResults.innerHTML = renderTopicPayload(payload);
      allQuestions.innerHTML = renderQuestionList(payload.questions || []);
      overviewQuestions.textContent = String(payload.totals?.questions ?? 0);
      overviewDocuments.textContent = String(payload.totals?.documents ?? 0);
    }

    async function loadStats() {
      const response = await fetch('/api/stats');
      const payload = await response.json();
      knownSubjects = payload.subjects || [];
      renderSubjectTabs();
      statsResults.innerHTML = renderStatsPayload(payload);
    }

    async function deleteSubject() {
      if (!currentSubject) {
        setStatus('Select a subject first.');
        return;
      }
      if (!window.confirm(`Delete subject "${currentSubject}" and all its uploaded files?`)) {
        return;
      }
      const response = await fetch(`/api/subjects/${encodeURIComponent(currentSubject)}`, {
        method: 'DELETE',
      });
      const payload = await response.json();
      setStatus(typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2));
      currentSubject = '';
      localStorage.removeItem('study.currentSubject');
      currentSubjectLabel.textContent = '-';
      subjectInput.value = '';
      emptyState.classList.remove('hidden');
      subjectDashboard.classList.add('hidden');
      allQuestions.innerHTML = renderEmpty('No extracted questions for this subject yet.');
      topicResults.innerHTML = renderEmpty('No topics found for this subject yet.');
      searchResults.innerHTML = renderEmpty('No matching material found.');
      await loadStats();
    }

    async function resetDatabase() {
      if (!window.confirm('Delete the entire study database and all uploaded files?')) {
        return;
      }
      const response = await fetch('/api/database/reset', { method: 'POST' });
      const payload = await response.json();
      setStatus(typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2));
      currentSubject = '';
      knownSubjects = [];
      localStorage.removeItem('study.currentSubject');
      currentSubjectLabel.textContent = '-';
      subjectInput.value = '';
      subjectHint.textContent = 'Create or select a subject tab to begin.';
      emptyState.classList.remove('hidden');
      subjectDashboard.classList.add('hidden');
      allQuestions.innerHTML = renderEmpty('No extracted questions for this subject yet.');
      topicResults.innerHTML = renderEmpty('No topics found for this subject yet.');
      searchResults.innerHTML = renderEmpty('No matching material found.');
      statsResults.innerHTML = renderEmpty('No stats loaded.');
      renderSubjectTabs();
      await loadStats();
    }

    document.getElementById('saveSubjectBtn').addEventListener('click', openSubjectFromInput);
    document.getElementById('uploadBtn').addEventListener('click', uploadFiles);
    document.getElementById('refreshSubjectBtn').addEventListener('click', loadSubjectOverview);
    document.getElementById('searchBtn').addEventListener('click', () => runSearch('search'));
    document.getElementById('questionBtn').addEventListener('click', () => runSearch('questions'));
    document.getElementById('deleteSubjectBtn').addEventListener('click', deleteSubject);
    document.getElementById('resetDbBtn').addEventListener('click', resetDatabase);
    subjectInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') openSubjectFromInput();
    });

    renderFiles();
    loadStats();
    if (currentSubject) {
      setCurrentSubject(currentSubject);
    }
  </script>
</body>
</html>
    """


@app.post("/api/ingest")
async def api_ingest(
    subject: str = Form(...),
    source_type: str = Form(...),
    vision_provider: str = Form("claude_cheap"),
    ocr_lang: str = Form("eng"),
    files: list[UploadFile] = File(...),
) -> JSONResponse:
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
                "topics": item.topics,
                "path": item.path,
                "text": item.chunk_text,
            }
            for item in results
        ],
    }


@app.get("/api/questions")
def api_questions(query: str, subject: str | None = None, limit: int = 10) -> dict:
    store = get_store()
    rows = store.question_search(query, limit=limit, subject=subject)
    return {
        "query": query,
        "subject": subject,
        "results": [
            {
                "subject": row["subject"],
                "source_type": row["source_type"],
                "document": row["name"],
                "page": row["page_number"],
                "question_number": row["question_number"],
                "topics": json.loads(row["topics_json"]),
                "path": row["path"],
                "text": row["text"],
            }
            for row in rows
        ],
    }


@app.get("/api/subject-overview")
def api_subject_overview(subject: str, question_limit: int = 120, topic_limit: int = 20) -> dict:
    store = get_store()
    overview = store.subject_overview(subject, question_limit=question_limit, topic_limit=topic_limit)
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
        "questions": [
            {
                "subject": row["subject"],
                "source_type": row["source_type"],
                "document": row["name"],
                "page": row["page_number"],
                "question_number": row["question_number"],
                "topics": json.loads(row["topics_json"]),
                "path": row["path"],
                "text": row["text"],
            }
            for row in overview["question_rows"]
        ],
    }


@app.get("/api/topics")
def api_topics(subject: str, limit: int = 15) -> dict:
    store = get_store()
    return {
        "subject": subject,
        "topics": [
            {"topic": topic, "count": count}
            for topic, count in store.subject_topics(subject, limit=limit)
        ],
    }


@app.get("/api/stats")
def api_stats() -> dict:
    store = get_store()
    return {
        "totals": store.stats(),
        "subjects": [dict(row) for row in store.subject_stats()],
    }


@app.delete("/api/subjects/{subject}")
def api_delete_subject(subject: str) -> dict:
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
def api_delete_subject_compat(payload: dict) -> dict:
    subject = str(payload.get("subject") or "").strip()
    return api_delete_subject(subject)


@app.post("/api/database/reset")
def api_reset_database() -> dict:
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
