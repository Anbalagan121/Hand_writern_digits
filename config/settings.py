"""Application configuration settings."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cnn_model.keras"
HISTORY_PATH = BASE_DIR / "data" / "history.json"
METRICS_PATH = BASE_DIR / "data" / "metrics.json"

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
MAX_FILE_SIZE_MB = 10

IMG_HEIGHT = 28
IMG_WIDTH = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10

CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
CANVAS_STROKE_WIDTH = 20
CANVAS_STROKE_COLOR = "#000000"
CANVAS_BG_COLOR = "#FFFFFF"
CANVAS_FILL_COLOR = "rgba(255, 255, 255, 1)"

PAGE_CONFIG = {
    "page_title": "Handwritten Digit Recognition",
    "page_icon": "✍️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

PAGES = [
    ("1_Home", "🏠 Home"),
    ("2_Digit_Prediction", "✍️ Digit Recognition"),
    ("3_Draw_Digit", "🎨 Draw Digit"),
    ("4_Image_Upload", "🖼️ Upload Image"),
    ("5_Dataset_Explorer", "📊 Dataset Explorer"),
    ("6_Model_Performance", "📈 Model Performance"),
    ("7_Model_Insights", "🧠 Model Insights"),
    ("8_About", "👨‍💻 About"),
]

DEVELOPER_INFO = {
    "name": "Anbalagan K",
    "role": "AI Engineer",
    "github": "https://github.com/Anbalagan121",
    "skills": [
        "Deep Learning",
        "Computer Vision",
        "Machine Learning",
        "Python",
        "TensorFlow/Keras",
        "PyTorch",
        "Streamlit",
        "MLOps",
        "Data Science",
    ],
    "technologies": [
        "Python",
        "TensorFlow/Keras",
        "PyTorch",
        "OpenCV",
        "NumPy",
        "Pandas",
        "Scikit-learn",
        "Matplotlib",
        "Seaborn",
        "Plotly",
        "Streamlit",
        "Docker",
        "Git",
    ],
    "deep_learning_tools": [
        "TensorFlow/Keras",
        "PyTorch",
        "Hugging Face Transformers",
        "OpenCV",
        "MLflow",
        "Weights & Biases",
    ],
    "project_description": (
        "A production-ready handwritten digit recognition web application "
        "built with Streamlit and TensorFlow/Keras. The application features "
        "a CNN trained on the MNIST dataset achieving 98.20% test accuracy. "
        "It includes interactive digit drawing, image upload, dataset exploration, "
        "model performance visualization, and architectural insights."
    ),
    "future_enhancements": [
        "Support for custom digit datasets",
        "Real-time webcam digit recognition",
        "Model comparison with multiple architectures",
        "SHAP/LIME explainability integration",
        "Batch prediction API endpoint",
        "Docker containerization and CI/CD pipeline",
        "Model versioning with MLflow",
        "User authentication and prediction history",
    ],
}