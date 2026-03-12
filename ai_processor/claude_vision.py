"""
Claude Vision: keep/drop classifier and question extraction from page images.
Uses Anthropic Messages API with base64 image input.
Cost-saving: resize images, lean prompts, Sonnet/Haiku, optional async batching.
"""

import base64
import io
import json
import os
from pathlib import Path

from anthropic import Anthropic
from PIL import Image

# Env: ANTHROPIC_API_KEY required
# CLAUDE_MODEL: claude-sonnet-4-20250514 (balance) or claude-3-5-haiku-20241022 (cheapest)
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CHEAP_MODEL = os.environ.get("CLAUDE_CHEAP_MODEL", "claude-3-5-haiku-20241022")
# Max dimension for resize; smaller = fewer tokens (Anthropic suggests ≤1568)
MAX_IMAGE_DIMENSION = int(os.environ.get("CLAUDE_MAX_IMAGE_DIMENSION", "1568"))
CLAUDE_ENABLED = os.environ.get("CLAUDE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}

KEEP_DROP_PROMPT = """You are classifying pages from a scanned exam PDF (PYQ – Previous Year Questions).

Look at this page image and decide whether it should be KEPT for question extraction or DROPPED.

KEEP the page (set "keep": true) if:
- It contains clearly readable PRINTED exam questions (typed text, not handwritten)
- It has question statements, sub-questions, or problem text that a student would answer
- The main content is printed/typed, even if there are small handwritten notes in margins

DROP the page (set "keep": false) if:
- The page is mostly or entirely handwritten (answers, notes, scribbles)
- The page is blank, a cover page, table of contents, or instruction sheet only
- The page has only diagrams/figures with no question text
- The page is too blurry, skewed, or damaged to read
- The page has only junk, noise, doodles, or illegible content

Reply with ONLY this JSON, nothing else: {"keep": true or false, "reason": "one short phrase"}"""


EXTRACT_QUESTIONS_PROMPT_TEMPLATE = """Extract only the PRINTED exam question statements from this page.

Do NOT extract:
- worked solutions
- answer keys
- method/procedure/explanation text
- derivation steps
- solved examples
- paragraphs beginning with Solution, Answer, Method, Steps, Working, Proof, or Explanation

If a page contains both a question and its solution, extract only the question statement and stop before the solution starts.

Important grouping rule:
- If one parent question has subparts like (a), (b), (c) or (i), (ii), (iii), return them as a single question entry.
- Do not split subparts into separate questions.
- Keep the parent question number as the question number.
- Include all subparts inside the same `text` field in the same order they appear on the page.

For each extracted question return: number (if visible), full question text verbatim (no summary), marks (if shown). No commentary. Ignore headers/footers/handwriting. If none, use empty "questions" array.

Reply with ONLY this JSON, nothing else:
{{"page": {page}, "questions": [{{"number": 1, "text": "full text here", "marks": 10}}, ...]}}"""


def _resize_image_if_needed(image: Image.Image, max_dim: int) -> Image.Image:
    """Resize so longest side is at most max_dim; preserve aspect ratio."""
    w, h = image.size
    if w <= max_dim and h <= max_dim:
        return image
    if w >= h:
        new_w, new_h = max_dim, int(h * max_dim / w)
    else:
        new_w, new_h = int(w * max_dim / h), max_dim
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _load_image_base64(image_path: str) -> tuple[str, str]:
    """Load image, resize to reduce tokens, return (base64_string, media_type)."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path).convert("RGB")
    img = _resize_image_if_needed(img, MAX_IMAGE_DIMENSION)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    return b64, "image/jpeg"


def _call_vision(prompt: str, image_path: str, max_tokens: int = 2048, model: str | None = None) -> str:
    """Send image + prompt to Claude, return assistant text."""
    if not CLAUDE_ENABLED:
        raise RuntimeError("Claude vision is disabled. Set CLAUDE_ENABLED=1 to enable it.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    b64, media_type = _load_image_base64(image_path)
    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model or MODEL,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    if not response.content:
        raise ValueError("Claude returned empty content")
    first = response.content[0]
    text = getattr(first, "text", None)
    if text is None:
        raise ValueError("Claude response has no text block")
    return text


def _parse_json_from_response(raw: str) -> dict:
    """Parse JSON from model output; strip markdown code blocks if present."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty response")
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def claude_keep_drop(image_path: str, model: str | None = None) -> dict:
    """
    Run keep/drop classifier on a page image.
    Returns {"keep": bool, "reason": str}.
    """
    raw = _call_vision(KEEP_DROP_PROMPT, image_path, max_tokens=128, model=model)
    obj = _parse_json_from_response(raw)
    keep = bool(obj.get("keep", False))
    reason = str(obj.get("reason", ""))
    return {"keep": keep, "reason": reason}


def claude_extract_questions(image_path: str, pdf: str, page: int, model: str | None = None) -> dict:
    """
    Extract structured questions from a page image.
    Returns {"pdf": str, "page": int, "page_id": str, "questions": [{"number", "text", "marks", "topic"}, ...]}.
    """
    prompt = EXTRACT_QUESTIONS_PROMPT_TEMPLATE.format(page=page)
    raw = _call_vision(prompt, image_path, max_tokens=3072, model=model)
    obj = _parse_json_from_response(raw)

    questions = obj.get("questions")
    if not isinstance(questions, list):
        questions = []

    # Normalize each question
    normalized = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        normalized.append({
            "number": q.get("number"),
            "text": (q.get("text") or "").strip(),
            "marks": q.get("marks"),
            "topic": q.get("topic"),
        })

    page_id = f"{pdf}::page_{page:03d}"
    return {
        "pdf": pdf,
        "page": page,
        "page_id": page_id,
        "image_path": image_path,
        "questions": normalized,
    }


def claude_extract_questions_cheap(image_path: str, pdf: str, page: int) -> dict:
    return claude_extract_questions(image_path, pdf, page, model=CHEAP_MODEL)
