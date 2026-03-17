from __future__ import annotations

import json
from collections import Counter
import re

from study_pipeline.models import Question
from study_pipeline.topics import canonicalize_topic, canonicalize_topics


def _post_json(base_url: str, model: str, prompt: str, timeout: int = 180) -> dict:
    import requests

    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        headers={"Content-Type": "application/json"},
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    raw_output = str(payload.get("response") or "").strip()
    if not raw_output:
        raise ValueError("Ollama returned an empty response while generating topics.")
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_output = "\n".join(lines).strip()
    return json.loads(raw_output)


def _prepare_examples(texts: list[str], limit: int = 80, max_chars: int = 320) -> list[str]:
    examples: list[str] = []
    seen: set[str] = set()
    for text in texts:
        compact = " ".join(str(text).split())
        if len(compact) < 20:
            continue
        compact = compact[:max_chars]
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        examples.append(compact)
        if len(examples) >= limit:
            break
    return examples


def syllabus_topics_from_text(text: str, limit: int = 40) -> list[str]:
    candidates: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip(" \t-*:.;")
        if not line:
            continue
        line = re.sub(r"^\(?[0-9ivxlcdm]+[.)-]?\s+", "", line, flags=re.I)
        line = re.sub(r"^(module|unit|chapter|topic)\s*[0-9ivxlcdm]*\s*[:.-]?\s*", "", line, flags=re.I)
        parts = [part.strip() for part in re.split(r"\s{2,}|[|/]|,\s(?=[A-Z])", line) if part.strip()]
        candidates.extend(parts or [line])
    return canonicalize_topics(candidates)[:limit]


def generate_subject_topic_catalog(
    subject: str,
    question_texts: list[str],
    base_url: str,
    model: str,
    existing_topics: list[str] | None = None,
    max_topics: int = 18,
) -> list[str]:
    examples = _prepare_examples(question_texts)
    if not examples:
        return canonicalize_topics(existing_topics or [])

    existing = canonicalize_topics(existing_topics or [])
    prompt = f"""
You are building a canonical PYQ topic list for the subject "{subject}".

Your job:
- Read the sample exam questions.
- Produce a stable list of meaningful exam topics.
- Merge synonyms and variants into one canonical label.
- Prefer noun phrases, not action words.
- Do not output boilerplate, generic verbs, marks, units, or random keywords.
- Reuse existing topic labels when they fit.
- Keep the list broad enough to group questions consistently.
- Return between 6 and {max_topics} topics.

Existing canonical topics to reuse when appropriate:
{json.dumps(existing, indent=2)}

Question samples:
{json.dumps(examples, indent=2)}

Return ONLY JSON:
{{
  "topics": ["Topic A", "Topic B"]
}}
""".strip()
    payload = _post_json(base_url=base_url, model=model, prompt=prompt)
    topics = canonicalize_topics([str(item) for item in payload.get("topics", [])])
    if existing:
        merged = canonicalize_topics(existing + topics)
        return merged[:max_topics]
    return topics[:max_topics]


def _fallback_primary_topic(question: Question, allowed_topics: list[str]) -> str | None:
    allowed = canonicalize_topics(allowed_topics)
    if not allowed:
        candidates = canonicalize_topics(question.topics)
        return candidates[0] if candidates else None

    allowed_keys = {topic.lower(): topic for topic in allowed}
    for topic in canonicalize_topics(question.topics):
        if topic.lower() in allowed_keys:
            return allowed_keys[topic.lower()]

    question_text = question.text.lower()
    scored: list[tuple[int, str]] = []
    for topic in allowed:
        score = 0
        for token in topic.lower().split():
            if len(token) >= 4 and token in question_text:
                score += 1
        if topic.lower() in question_text:
            score += 3
        if score:
            scored.append((score, topic))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return scored[0][1] if scored else allowed[0]


def assign_primary_topics(
    subject: str,
    questions: list[Question],
    allowed_topics: list[str],
    base_url: str,
    model: str,
) -> list[Question]:
    canonical_topics = canonicalize_topics(allowed_topics)
    if not questions:
        return questions
    if not canonical_topics:
        for question in questions:
            question.primary_topic = _fallback_primary_topic(question, [])
        return questions

    questions_payload = [
        {
            "index": index,
            "question_number": question.question_number,
            "text": " ".join(question.text.split())[:700],
        }
        for index, question in enumerate(questions, start=1)
    ]
    prompt = f"""
You are classifying previous year exam questions for the subject "{subject}".

Allowed topics:
{json.dumps(canonical_topics, indent=2)}

Rules:
- Assign exactly one topic to each question.
- Use only one topic from the allowed list.
- Do not create new topics.
- Choose the most central concept being tested.
- Be consistent across similar questions.

Questions:
{json.dumps(questions_payload, indent=2)}

Return ONLY JSON:
{{
  "assignments": [
    {{"index": 1, "primary_topic": "One Allowed Topic"}}
  ]
}}
""".strip()

    assignments: dict[int, str] = {}
    try:
        payload = _post_json(base_url=base_url, model=model, prompt=prompt)
        for item in payload.get("assignments", []):
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", 0))
            primary_topic = canonicalize_topic(str(item.get("primary_topic") or ""))
            if not primary_topic:
                continue
            match = next((topic for topic in canonical_topics if topic.lower() == primary_topic.lower()), None)
            if match:
                assignments[index] = match
    except Exception:
        assignments = {}

    for index, question in enumerate(questions, start=1):
        question.primary_topic = assignments.get(index) or _fallback_primary_topic(question, canonical_topics)
    return questions


def infer_primary_topics(
    subject: str,
    questions: list[Question],
    existing_question_texts: list[str],
    existing_topics: list[str],
    syllabus_text: str | None,
    base_url: str,
    model: str,
) -> tuple[list[str], list[Question]]:
    syllabus_topics = syllabus_topics_from_text(syllabus_text or "")
    if syllabus_topics:
        return syllabus_topics, assign_primary_topics(
            subject=subject,
            questions=questions,
            allowed_topics=syllabus_topics,
            base_url=base_url,
            model=model,
        )

    question_texts = existing_question_texts + [question.text for question in questions]
    try:
        catalog = generate_subject_topic_catalog(
            subject=subject,
            question_texts=question_texts,
            base_url=base_url,
            model=model,
            existing_topics=existing_topics,
        )
    except Exception:
        catalog = canonicalize_topics(existing_topics)

    if not catalog:
        extracted = []
        for question in questions:
            extracted.extend(question.topics)
        counts = Counter(canonicalize_topics(extracted))
        catalog = [topic for topic, _ in counts.most_common(12)]

    return catalog, assign_primary_topics(
        subject=subject,
        questions=questions,
        allowed_topics=catalog,
        base_url=base_url,
        model=model,
    )
