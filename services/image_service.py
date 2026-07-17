"""Image processing service."""

import streamlit as st
from PIL import Image, ImageOps
import numpy as np

from config.settings import IMG_HEIGHT, IMG_WIDTH
from utils.exceptions import ImageProcessingError
from utils.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_data(show_spinner="Preprocessing image...")
def preprocess_uploaded_image(image: Image.Image) -> tuple:
    """
    Preprocess uploaded image for MNIST model.
    
    Returns:
        tuple: (model_input_array, preview_array)
    """
    try:
        img_gray = image.convert("L")
        img_inv = ImageOps.invert(img_gray)
        img_resized = img_inv.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        model_input = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        preview = img_array
        return model_input, preview
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ImageProcessingError(f"Failed to preprocess image: {e}") from e


@st.cache_data(show_spinner="Preprocessing canvas drawing...")
def preprocess_canvas_image(canvas_data) -> tuple:
    """
    Preprocess canvas drawing for MNIST model.
    
    Returns:
        tuple: (model_input_array, preview_array)
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
        model_input = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        preview = img_array
        return model_input, preview
    except Exception as e:
        logger.error(f"Canvas preprocessing failed: {e}")
        raise ImageProcessingError(f"Failed to preprocess canvas: {e}") from e


def get_image_info(image: Image.Image) -> dict:
    """Extract image metadata."""
    return {
        "format": image.format,
        "size": image.size,
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
    }