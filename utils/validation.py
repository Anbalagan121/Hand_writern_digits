"""Input validation utilities."""

import os
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps
import numpy as np

from config.settings import SUPPORTED_IMAGE_TYPES, MAX_FILE_SIZE_MB, IMG_HEIGHT, IMG_WIDTH
from utils.exceptions import InvalidImageError, ImageProcessingError


def validate_image_file(file) -> None:
    """Validate uploaded image file."""
    if file is None:
        raise InvalidImageError("No file uploaded")

    file.seek(0, os.SEEK_END)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if size_mb > MAX_FILE_SIZE_MB:
        raise InvalidImageError(f"File size ({size_mb:.1f}MB) exceeds limit ({MAX_FILE_SIZE_MB}MB)")

    ext = Path(file.name).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_IMAGE_TYPES:
        raise InvalidImageError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_IMAGE_TYPES}")


def validate_image_content(image: Image.Image) -> None:
    """Validate image content after opening."""
    if image is None:
        raise InvalidImageError("Failed to open image")

    valid_modes = ("L", "RGB", "RGBA", "P")
    if image.mode not in valid_modes:
        raise InvalidImageError(f"Unsupported image mode: {image.mode}. Use grayscale or RGB.")


def validate_canvas_data(canvas_data) -> bool:
    """Check if canvas has any drawing."""
    if canvas_data is None or canvas_data.image_data is None:
        return False
    data = canvas_data.image_data[:, :, :3]
    return bool(np.any(data < 255))


def preprocess_for_mnist(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image to match MNIST training format.
    
    MNIST format: 28x28, grayscale, white digit on black background, normalized 0-1.
    
    Returns:
        tuple: (processed_array_for_model, preview_array_for_display)
    """
    try:
        img_gray = image.convert("L")
        img_inv = ImageOps.invert(img_gray)
        img_resized = img_inv.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        processed = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        preview = img_array
        return processed, preview
    except Exception as e:
        raise ImageProcessingError(f"Failed to preprocess image: {e}") from e


def preprocess_canvas_for_mnist(canvas_data) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess canvas drawing to match MNIST format.
    
    Canvas returns RGBA with white background and black strokes.
    """
    try:
        from PIL import Image
        img = Image.fromarray(canvas_data.astype("uint8"), "RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img_gray = background.convert("L")
        img_inv = ImageOps.invert(img_gray)
        img_resized = img_inv.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        processed = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        preview = img_array
        return processed, preview
    except Exception as e:
        raise ImageProcessingError(f"Failed to preprocess canvas: {e}") from e