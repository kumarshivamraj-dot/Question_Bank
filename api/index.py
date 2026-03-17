from __future__ import annotations

import os
from pathlib import Path


TMP_ROOT = Path("/tmp/study-index")
TMP_ROOT.mkdir(parents=True, exist_ok=True)
(TMP_ROOT / "uploads").mkdir(parents=True, exist_ok=True)
(TMP_ROOT / "processed").mkdir(parents=True, exist_ok=True)
(TMP_ROOT / "original_views").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("STUDY_UPLOAD_ROOT", str(TMP_ROOT / "uploads"))
os.environ.setdefault("STUDY_DB_PATH", str(TMP_ROOT / "processed" / "study_index.db"))
os.environ.setdefault("STUDY_ORIGINAL_ASSET_ROOT", str(TMP_ROOT / "original_views"))
os.environ.setdefault("STUDY_EMBEDDING_PROVIDER", "hashing")

from study_web.app import app

