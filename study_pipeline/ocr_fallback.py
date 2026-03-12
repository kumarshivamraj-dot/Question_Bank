from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(grey, h=20)
    threshold = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )

    coords = np.column_stack(np.where(threshold < 255))
    if coords.size:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        height, width = threshold.shape
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        threshold = cv2.warpAffine(
            threshold,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return Image.fromarray(threshold)


def ocr_image(image: Image.Image, lang: str = "eng") -> str:
    processed = preprocess_image_for_ocr(image)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        processed.save(temp_path)
        return pytesseract.image_to_string(
            Image.open(temp_path), config="--oem 3 --psm 6", lang=lang
        )
    finally:
        temp_path.unlink(missing_ok=True)

