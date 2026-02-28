import json
import os
import requests

# In Docker, .cursor is mounted at /app/.cursor; log path must be container path
LOG_PATH = os.environ.get("DEBUG_LOG_PATH", "/app/.cursor/debug-c5afe7.log")


def _debug_log(payload):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def extract_topics_from_ocr(ocr_obj):
    """Extract important topics/subjects from messy PYQ OCR text."""
    prompt = f"""
You are a PYQ (Previous Year Question) analyzer.

You will receive messy OCR text from an exam PDF page. Extract the IMPORTANT TOPICS or SUBJECTS covered in that page.
- Focus on concepts, topics, formulas, or subject areas mentioned (e.g. "Newton's Laws", "Recursion", "SQL Joins", "Integration by Parts")
- Ignore OCR noise, question numbers, and boilerplate
- Return only topics that appear to be actual exam content
- Be concise: 2-8 topics per page is typical

Return ONLY valid JSON in this format:

{{
  "page": <page number>,
  "topics": ["topic1", "topic2", "topic3"]
}}

If the page has no discernible topics, use an empty array: {{"page": N, "topics": []}}

Here is the input:
{json.dumps(ocr_obj, indent=2)}
""".strip()

    base_url = os.environ["OLLAMA_BASE_URL"].rstrip("/")

    response = requests.post(
        f"{base_url}/api/generate",
        headers={"Content-Type": "application/json"},
        json={"model": "llama3", "prompt": prompt, "stream": False, "format": "json"},
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    # #region agent log
    _debug_log({"sessionId": "c5afe7", "hypothesisId": "A,D,E", "location": "ollama.py:post-response", "message": "Ollama API response structure", "data": {"data_keys": list(data.keys()) if isinstance(data, dict) else "not_dict", "has_response": "response" in data if isinstance(data, dict) else False, "response_type": type(data.get("response")).__name__ if isinstance(data, dict) else "n/a", "response_len": len(str(data.get("response", ""))) if isinstance(data, dict) else 0, "response_preview": (str(data.get("response", ""))[:500] if isinstance(data, dict) else ""), "response_repr": repr(data.get("response", ""))[:300] if isinstance(data, dict) else ""}, "timestamp": __import__("time").time() * 1000})
    # #endregion

    raw_output = (data.get("response") or "").strip()

    # #region agent log
    _debug_log({"sessionId": "c5afe7", "hypothesisId": "B,C", "location": "ollama.py:pre-json-loads", "message": "raw_output before json.loads", "data": {"raw_output_len": len(raw_output) if raw_output else 0, "raw_output_empty": raw_output == "" or raw_output is None, "raw_output_first_400": (raw_output or "")[:400], "raw_output_repr": repr(raw_output)[:200]}, "timestamp": __import__("time").time() * 1000})
    # #endregion

    if not raw_output:
        raise ValueError(
            "Ollama returned an empty response. "
            "Ensure 'llama3' is pulled: run 'ollama pull llama3' in the ollama container or host."
        )

    # Strip markdown code blocks if the model wrapped JSON in ```json ... ```
    if raw_output.startswith("```"):
        lines = raw_output.strip().split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_output = "\n".join(lines)

    return json.loads(raw_output)
