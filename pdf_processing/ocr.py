from PIL import Image
import cv2
import numpy as np
import os
import pytesseract


def preProcessor(imagePath: str, out_path: str):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError("Image doesn't exists")

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(grey, h=30)

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernal)
    normalization = cv2.divide(denoised, bg, scale=255)

    sharpned_kernal = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpned = cv2.filter2D(normalization, -1, sharpned_kernal)

    # Adaptative threshold
    thresh = cv2.adaptiveThreshold(
        sharpned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # de skew

    coords = np.column_stack(np.where(thresh < 255))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = thresh.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    deskewed = cv2.warpAffine(
        thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    if not cv2.imwrite(out_path, deskewed):
        raise IOError(f"Failed to write output image: {out_path}")

    return out_path


def run_tessaract(destpath):
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(Image.open(destpath), config=config, lang="eng")
