"""Data loading service."""

import json
import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist

from config.settings import HISTORY_PATH, METRICS_PATH
from utils.exceptions import DataLoadError
from utils.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_data(show_spinner="Loading MNIST dataset...")
def load_mnist():
    """Load MNIST dataset with caching."""
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        logger.info(f"MNIST loaded: train={x_train.shape}, test={x_test.shape}")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logger.error(f"MNIST loading failed: {e}")
        raise DataLoadError(f"Failed to load MNIST: {e}") from e


@st.cache_data(show_spinner="Loading training history...")
def load_training_history():
    """Load training history from JSON."""
    try:
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"History loading failed: {e}")
        raise DataLoadError(f"Failed to load history: {e}") from e


@st.cache_data(show_spinner="Loading test metrics...")
def load_test_metrics():
    """Load test metrics and confusion matrix."""
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Metrics loading failed: {e}")
        raise DataLoadError(f"Failed to load metrics: {e}") from e


def get_class_distribution(labels: np.ndarray):
    """Calculate class distribution."""
    return np.unique(labels, return_counts=True)


def get_sample_indices(labels: np.ndarray, num_classes: int = 10):
    """Get first occurrence index for each class."""
    return [np.where(labels == i)[0][0] for i in range(num_classes)]