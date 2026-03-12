from pdf_converter import convert_pdf_to_images
from ocr import preProcessor, run_tessaract
from post_process import post_processing
from page_classifier import (
    classify_page_heuristic,
    make_page_id,
    make_image_path,
)
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_processor.claude_vision import claude_keep_drop, claude_extract_questions
import json
import os
import shutil
import unicodedata


pdf_path = Path("/data/input")
output_path = Path("/data/pngs")
processed_pngs_path = Path("/data/processing")  # directory for processed images


def cleanup_run_dirs(*dirs: Path) -> None:
    """Clear contents of the given directories so each run starts from a clean state.
    Removes only contents, not the directory itself, to avoid 'directory is busy' on
    Docker volume mounts."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        for entry in os.listdir(d):
            path = d / entry
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


# Reset pngs and processing at the start of every run
cleanup_run_dirs(output_path, processed_pngs_path)

all_pages_ocr_text = []
Final_ocr = []
processed_dirs_this_run = []


# Each PDF in the path is converted into pages and then the pages are preprocessed
for pdf in pdf_path.glob("*.pdf"):
    pdf_stem = pdf.stem
    pdf_output_dir = output_path / pdf_stem
    pdf_processed_dir = processed_pngs_path / pdf_stem
    os.makedirs(pdf_output_dir, exist_ok=True)
    os.makedirs(pdf_processed_dir, exist_ok=True)
    processed_dirs_this_run.append(pdf_processed_dir)

    image_list_path = convert_pdf_to_images(pdf, pdf_output_dir)  # List[Path]

    for i, image_path in enumerate(image_list_path, start=1):
        processed_image_path = pdf_processed_dir / f"page_{i:03d}.png"
        preProcessor(str(image_path), str(processed_image_path))

# Converting the processed images into string (only files from this run)
for pdf_processed_dir in processed_dirs_this_run:
    for page, final_image in enumerate(
        sorted(pdf_processed_dir.glob("*.png")), start=1
    ):
        ocr_text = run_tessaract(final_image)
        ocr_text = unicodedata.normalize("NFKC", ocr_text)
        all_pages_ocr_text.append(
            {"pdf": pdf_processed_dir.name, "page": page, "raw_text": ocr_text}
        )


# Making the List[str] using the list loop .. This would replace the i with the current iteration and make a list of the all the strings
raw_text = [p["raw_text"] for p in all_pages_ocr_text]
cleaned_text = post_processing(raw_text)
for i, text in enumerate(cleaned_text):
    Final_ocr.append(
        {
            "pdf": all_pages_ocr_text[i]["pdf"],
            "page": all_pages_ocr_text[i]["page"],
            "cleaned_text": text,
        }
    )

# Classify each page (question / no_question / maybe_question) using heuristics
# Add canonical page_id and image_path for consistent lookup when sending to Claude
processed_dir_str = str(processed_pngs_path)
Final_classified = []
for obj in Final_ocr:
    pdf = obj["pdf"]
    page = obj["page"]
    text = obj["cleaned_text"]

    classification = classify_page_heuristic(text)
    page_id = make_page_id(pdf, page)
    image_path = make_image_path(processed_dir_str, pdf, page)

    Final_classified.append(
        {
            "pdf": pdf,
            "page": page,
            "page_id": page_id,
            "image_path": image_path,
            "classification": classification,
            "cleaned_text": text,
        }
    )

# Save page classifications (consistent page_id used everywhere)
classification_file = Path("/data/processed/page_classifications.json")
classification_file.parent.mkdir(parents=True, exist_ok=True)
summary = {
    "question": sum(1 for p in Final_classified if p["classification"] == "question"),
    "no_question": sum(
        1 for p in Final_classified if p["classification"] == "no_question"
    ),
    "maybe_question": sum(
        1 for p in Final_classified if p["classification"] == "maybe_question"
    ),
}
classification_data = {
    "by_page": Final_classified,
    "summary": summary,
}
with open(classification_file, "w") as f:
    json.dump(classification_data, f, indent=2)
print(json.dumps(classification_data, indent=2))
print(f"\nPage classifications saved to {classification_file}")

# Send images to Claude by classification (batch/concurrent to reduce latency and cost):
# - "question" -> direct to Claude question extraction
# - "maybe_question" -> Claude keep/drop first; if keep, then question extraction
# - "no_question" -> skip (do not send to Claude)
def _process_page_for_claude(obj):
    pdf = obj["pdf"]
    page = obj["page"]
    page_id = obj["page_id"]
    image_path = obj["image_path"]
    classification = obj["classification"]

    if classification == "no_question":
        return None

    try:
        if classification == "maybe_question":
            keep_drop = claude_keep_drop(image_path)
            if not keep_drop.get("keep"):
                print(f"  [Claude keep/drop] {page_id} DROP: {keep_drop.get('reason', '')}")
                return None
            print(f"  [Claude keep/drop] {page_id} KEEP: {keep_drop.get('reason', '')}")

        out = claude_extract_questions(image_path, pdf, page)
        out["classification"] = classification
        return out
    except Exception as exc:
        print(f"  [Claude error] {page_id} SKIP: {exc}")
        return None

pages_to_process = [obj for obj in Final_classified if obj["classification"] != "no_question"]
max_workers = int(os.environ.get("CLAUDE_MAX_CONCURRENT", "4"))
question_results = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_process_page_for_claude, obj): obj for obj in pages_to_process}
    for future in as_completed(futures):
        try:
            result = future.result()
        except Exception as exc:
            print(f"  [Claude worker error] {exc}")
            result = None
        if result is not None:
            question_results.append(result)

# Sort by pdf then page so output order is stable
question_results.sort(key=lambda r: (r["pdf"], r["page"]))

# Flatten for convenience
questions_flat = []
for r in question_results:
    for q in r.get("questions", []):
        questions_flat.append(
            {
                "pdf": r["pdf"],
                "page": r["page"],
                "page_id": r["page_id"],
                "image_path": r["image_path"],
                **q,
            }
        )

# Save questions JSON (consistent page_id throughout)
questions_file = Path("/data/processed/questions.json")
questions_data = {
    "by_page": question_results,
    "questions_flat": questions_flat,
}
with open(questions_file, "w") as f:
    json.dump(questions_data, f, indent=2)
print(json.dumps(questions_data, indent=2))
print(f"\nQuestions saved to {questions_file}")
