"""Prediction service."""

import numpy as np
import streamlit as st

from utils.exceptions import PredictionError
from utils.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_data(show_spinner="Running prediction...")
def predict_digit(model, processed_image: np.ndarray) -> dict:
    """
    Run prediction on preprocessed image.
    
    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array (1, 28, 28, 1)
        
    Returns:
        dict: Prediction results
    """
    try:
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class]) * 100
        
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = [
            {
                "digit": int(idx),
                "probability": float(predictions[idx]),
                "percentage": float(predictions[idx]) * 100,
            }
            for idx in top_3_indices
        ]
        
        return {
            "predicted_digit": predicted_class,
            "confidence": confidence,
            "all_probabilities": predictions.tolist(),
            "top_3": top_3,
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise PredictionError(f"Prediction failed: {e}") from e


def format_prediction_results(result: dict) -> dict:
    """Format prediction results for display."""
    return {
        "main": str(result["predicted_digit"]),
        "confidence": f"{result['confidence']:.2f}%",
        "probabilities": result["all_probabilities"],
        "top_3": result["top_3"],
    }