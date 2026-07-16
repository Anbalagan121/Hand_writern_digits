import os
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_cnn_model(model_path="models/cnn_model.keras"):
    """
    Load the trained CNN model from disk.
    Cached so it's only loaded once per session.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, model_path)
    
    if os.path.exists(full_path):
        return load_model(full_path)
    else:
        raise FileNotFoundError(f"Model file not found at {full_path}. Please train the model first.")
