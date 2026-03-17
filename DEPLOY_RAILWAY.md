# Railway Deploy

This repo can run on Railway without changing the local workflow.

## Recommended Railway setup

Create a Railway project and deploy this repo as a Python web service.

Use these environment variables:

```bash
STUDY_UPLOAD_ROOT=/data/uploads
STUDY_DB_PATH=/data/processed/study_index.db
STUDY_ORIGINAL_ASSET_ROOT=/data/processed/original_views
STUDY_EMBEDDING_PROVIDER=hashing
STUDY_ADMIN_KEY=your-secret-admin-key
```

Optional:

```bash
STUDY_TOPIC_OLLAMA_MODEL=llama3
STUDY_OLLAMA_MODEL=nomic-embed-text
STUDY_OLLAMA_URL=http://localhost:11434
STUDY_TOPIC_OLLAMA_URL=http://localhost:11434
```

## Persistent storage

Attach a Railway volume and mount it at:

```bash
/data
```

This is important because the app stores:

- uploaded files
- SQLite database
- generated original-view PNGs

Without a mounted volume, your data will be lost on redeploy or restart.

## Start command

Railway can use either:

```bash
uvicorn study_web.app:app --host 0.0.0.0 --port $PORT
```

or the included `Procfile`.

## Notes

- This app is a better fit for Railway than Vercel because it needs persistent filesystem storage.
- Public viewing works without a key.
- Admin actions require `STUDY_ADMIN_KEY` if it is set.
