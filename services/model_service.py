"""Model loading and management service."""

import io
import os
from contextlib import redirect_stdout

import numpy as np
import streamlit as st

from config.settings import MODEL_PATH
from utils.exceptions import ModelLoadError
from utils.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_resource(show_spinner="Loading AI model...")
def load_model_cached(model_path: str = None) -> object:
    """
    Load trained CNN model with caching.
    
    Args:
        model_path: Optional custom model path
        
    Returns:
        Loaded Keras model
    """
    path = model_path or str(MODEL_PATH)
    
    if not os.path.exists(path):
        logger.error(f"Model not found at {path}")
        raise ModelLoadError(
            f"Model file not found at {path}. "
            "Please train the model first using train_and_save.py"
        )
    
    try:
        from tensorflow.keras.models import load_model

        model = load_model(path)
        logger.info(f"Model loaded: {model.name}, params={model.count_params():,}")
        return model
    except ImportError as exc:
        logger.error(f"TensorFlow is unavailable in this runtime: {exc}")
        raise ModelLoadError(
            "TensorFlow is not available in the current runtime. Please use a Python 3.11 environment with tensorflow-cpu installed."
        ) from exc
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise ModelLoadError(f"Failed to load model: {e}") from e


def get_model_summary(model) -> str:
    """Get model architecture summary as string."""
    import io
    from contextlib import redirect_stdout
    
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    return stream.getvalue()


def get_model_info(model) -> dict:
    """Extract model information."""
    total_params = model.count_params()
    trainable_params = sum(
        np.prod(w.shape) for w in model.trainable_weights
    )
    non_trainable = total_params - trainable_params
    
    return {
        "name": model.name,
        "layers": len(model.layers),
        "total_params": total_params,
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }