# Study Index

This repo now contains a practical local study system instead of an OCR-only PDF pipeline.

The default workflow is:

1. Ingest previous-year papers, reference books, class notes, PPTs, and handouts.
2. Extract native PDF text first.
3. Use OCR only on pages that do not contain usable embedded text.
4. Chunk the content and store it in one SQLite database.
5. Save extracted questions and topic hints alongside the chunks.
6. Search everything from one place with a local vector store.
7. Keep everything grouped by subject inside the same database.

## Why this is better

The old flow converted every page into an image, OCRed everything, and only then tried to recover structure. That is the wrong default for academic PDFs because:

- normal PDFs already contain text
- OCR should be fallback, not the primary path
- question extraction works better after text-first cleanup
- one unified database is more useful than scattered JSON files

## What is included

The new implementation lives in `study_pipeline/`.

- `extract.py`: text-first document extraction with OCR fallback
- `chunking.py`: chunking for retrieval
- `questions.py`: exam-question extraction heuristics
- `topics.py`: lightweight topic hints from headings and repeated terms
- `embeddings.py`: vector embeddings
- `store.py`: SQLite database with chunks, questions, FTS, and vectors
- `cli.py`: command-line entrypoint

## Database contents

The SQLite database stores:

- documents
- chunks for retrieval
- extracted questions
- topic labels per chunk/question
- embedding vectors per chunk
- subject name per document
- source type per document (`pyq`, `reference`, `handout`, `notes`, `ppt`)

Default location:

```bash
data/processed/study_index.db
```

## Embedding modes

Two modes are supported:

- `hashing`: fully local, no model download, good enough as a baseline
- `ollama`: better semantic retrieval if you already run Ollama embeddings

If you use Ollama, a practical model is `nomic-embed-text`.

## Supported files

- `.pdf`
- `.json`
- `.txt`
- `.md`
- `.pptx`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For OCR fallback you also need Tesseract installed on your machine.

## Usage

Index a previous-year paper and a set of notes under a subject:

```bash
python -m study_pipeline.cli ingest --subject dbms --source-type pyq data/input/ilovepdf_merged-22.pdf
python -m study_pipeline.cli ingest --subject dbms --source-type reference notes/unit1.md slides/unit2.pptx
```

Use a vision model for PYQs:

```bash
python -m study_pipeline.cli ingest --subject dbms --source-type pyq --vision-provider claude data/input/ilovepdf_merged-22.pdf
```

Use Ollama embeddings instead of the default hashing store:

```bash
python -m study_pipeline.cli --embedding-provider ollama --ollama-model nomic-embed-text ingest --subject dbms --source-type pyq data/input/ilovepdf_merged-22.pdf
```

Search across all indexed material:

```bash
python -m study_pipeline.cli search "laplace transform properties" --subject dbms
```

Search only extracted questions:

```bash
python -m study_pipeline.cli questions "binary tree" --subject dsa
```

Show important topics for one subject:

```bash
python -m study_pipeline.cli topics --subject dbms
```

Show database stats:

```bash
python -m study_pipeline.cli stats
```

## Web App

If you are running this in Docker, do not start `uvicorn` manually inside a container and expect host access unless the port is published and the server binds to `0.0.0.0`.

Use the dedicated web service instead:

```bash
docker compose up --build study-web
```

Then open:

```text
http://127.0.0.1:8000
```

The `study-web` container:

- publishes `8000:8000` to the host
- runs `uvicorn study_web.app:app --host 0.0.0.0 --port 8000`
- shares `./data` with the host so uploads and the SQLite database persist
- can talk to Ollama over the internal Docker network
- keeps `CLAUDE_ENABLED=0` by default so the app does not crash when external Anthropic access is unavailable

If you are not using Docker, this still works locally:

```bash
uvicorn study_web.app:app --reload --host 0.0.0.0 --port 8000
```

If you really want Claude vision inside Docker, enable it explicitly:

```bash
CLAUDE_ENABLED=1 docker compose up --build study-web
```

The UI lets you:

- drag and drop PDFs, PPTXs, notes, and handouts
- drag and drop structured question JSON files
- assign a `subject` and `source type`
- use Claude vision for scanned PYQs when you want
- search all indexed material for a subject
- get questions for a topic
- get recurring important topics for a subject
- delete one subject or wipe the full database from the UI

Uploaded files are stored under:

```bash
data/uploads/<subject>/<source_type>/
```

## JSON Upload Format

If you already have structured questions, upload a `.json` file instead of a PDF. That is the cleaner path.

Accepted shape:

```json
{
  "questions": [
    {
      "question_number": "1",
      "text": "Explain normalization.",
      "subparts": [
        { "label": "a", "text": "Define 1NF." },
        { "label": "b", "text": "Define 2NF." }
      ],
      "topics": ["normalization", "1NF", "2NF"],
      "page": 1
    }
  ]
}
```

Also accepted:

```json
[
  {
    "question_number": "1",
    "text": "Explain normalization."
  }
]
```

## Practical notes

- Put all your academic material into the same database, but always ingest with the correct `--subject`.
- Re-run `ingest` on the same file when the source changes. The database upserts by file path.
- Use `search` for concept revision, `questions` for exam practice, and `topics` to see what keeps recurring inside a subject.
- The hashing embedding mode is intentionally dependency-light. If you want stronger semantic search, switch to Ollama embeddings.
- For PYQs, `--vision-provider claude` is the intended high-quality path when you want question extraction from scanned papers.

## Next sensible improvements

- add a small web UI on top of the SQLite database
- add syllabus/unit tagging
- add flashcard generation from high-frequency questions
- add answer synthesis over top retrieved chunks
