from pdf_converter import convert_pdf_to_images
from ocr import preProcessor, run_tessaract
from post_process import post_processing
from pathlib import Path
from ai_processor.ollama import extract_topics_from_ocr
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
    for page, final_image in enumerate(sorted(pdf_processed_dir.glob("*.png")), start=1):
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

# Extract topics from each page using the LLM
results = []
for obj in Final_ocr:
    out = extract_topics_from_ocr(obj)
    out["pdf"] = obj["pdf"]
    results.append(out)

# Save to JSON for use downstream (n8n, tagging, analysis)
output_file = Path("/data/processed/topics.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

all_topics = []
for r in results:
    all_topics.extend(r.get("topics", []))
output_data = {"by_page": results, "all_topics_unique": sorted(set(all_topics))}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(json.dumps(output_data, indent=2))
print(f"\nTopics saved to {output_file}")
