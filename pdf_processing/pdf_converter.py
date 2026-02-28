"""
pdf_converter.py
Handles PDF to image conversion
"""

from pathlib import Path
from pdf2image import convert_from_path
import os
from typing import List


"""Convert a single PDF into PNG images in the given output directory.

Args:
    pdf_path: Path to a single PDF file.
    output_path: Directory where PNG pages will be written.

Returns:
    A list of Paths to the generated PNG images.
"""


def convert_pdf_to_images(pdf_path: Path, output_path: Path) -> List[Path]:
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # pdf2image will write PNGs into output_path and return Image objects
    images = convert_from_path(
        pdf_path,
        dpi=300,
        output_folder=output_path,
        fmt="png",
        thread_count=3,
    )

    # Save pages with a predictable name pattern and collect their paths
    output_paths: List[Path] = []
    for i, image in enumerate(images, start=1):
        filename = output_path / f"page_{i:03d}.png"
        image.save(filename)
        output_paths.append(filename)

    return output_paths
