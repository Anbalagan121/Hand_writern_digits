"""Main application entry point - Home page."""

import streamlit as st
from config.settings import PAGE_CONFIG, DEVELOPER_INFO
from utils.logging_config import setup_logging

# Initialize logging
setup_logging()

# Page config
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for professional look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 500;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        color: #1a1a2e;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Home page content."""
    st.title("🏠 Handwritten Digit Recognition")
    st.markdown("### Production-ready AI web application powered by CNN and Streamlit")
    st.markdown("---")

    # Hero section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Project Overview

        Welcome to the **Handwritten Digit Recognition** AI Web Application! This production-ready 
        dashboard showcases a complete end-to-end Deep Learning workflow for recognizing handwritten 
        digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

        Built with professional software engineering practices: modular architecture, caching, 
        error handling, logging, and responsive UI design.
        """)

        st.markdown("### 🎯 Business Problem")
        st.info("""
        Accurate recognition of handwritten digits is crucial in various automated systems:
        - **Postal mail sorting**: Reading ZIP codes on envelopes
        - **Bank check processing**: Automating check amount reading
        - **Form digitization**: Converting handwritten forms to digital data
        - **License plate recognition**: Traffic monitoring systems
        """)

        st.markdown("### 🎯 Objectives")
        obj_col1, obj_col2, obj_col3 = st.columns(3)
        with obj_col1:
            st.metric("Test Accuracy", "98.20%", "Target: >98%")
        with obj_col2:
            st.metric("Inference Time", "< 50ms", "Real-time")
        with obj_col3:
            st.metric("Model Size", "8.4 MB", "CPU Optimized")

    with col2:
        st.markdown("### 🛠️ Technology Stack")
        tech_stack = {
            "Frontend": "Streamlit",
            "Deep Learning": "TensorFlow/Keras",
            "Data Processing": "NumPy, Pandas, Pillow",
            "Visualization": "Plotly, Seaborn",
            "ML Ops": "Scikit-learn",
        }
        for key, value in tech_stack.items():
            st.markdown(f"- **{key}**: {value}")

        st.markdown("### ✨ Key Features")
        features = [
            "✍️ **Digit Recognition** - Upload images for instant prediction",
            "🎨 **Interactive Canvas** - Draw digits with mouse/touch",
            "🖼️ **Image Upload** - PNG, JPG, JPEG with preprocessing view",
            "📊 **Dataset Explorer** - MNIST statistics & samples",
            "📈 **Model Performance** - Interactive charts & confusion matrix",
            "🧠 **Model Insights** - Architecture & prediction pipeline",
            "👨‍💻 **About** - Developer info & project details",
        ]
        for feature in features:
            st.markdown(feature)

    st.markdown("---")

    # Deep Learning Workflow
    st.markdown("### 🔬 Deep Learning Workflow")
    workflow_steps = [
        ("1. Data Collection", "Load MNIST dataset (60K train + 10K test)"),
        ("2. Preprocessing", "Normalize, reshape to 28×28×1, invert colors"),
        ("3. Model Building", "CNN: Conv2D → MaxPool → Flatten → Dense → Dense"),
        ("4. Training", "5 epochs, Adam optimizer, sparse categorical crossentropy"),
        ("5. Evaluation", "Test on 10K unseen images: 98.20% accuracy"),
        ("6. Deployment", "Package as Streamlit web application"),
    ]

    for i, (step, desc) in enumerate(workflow_steps):
        col_step, col_desc = st.columns([1, 4])
        with col_step:
            st.markdown(f"**{step}**")
        with col_desc:
            st.markdown(desc)

    st.markdown("---")

    # CNN Architecture
    with st.expander("🏗️ CNN Architecture Details", expanded=False):
        st.markdown("""
        **Model: Sequential CNN**
        
        | Layer | Type | Output Shape | Parameters |
        |-------|------|--------------|------------|
        | 1 | Conv2D (32 filters, 3×3, ReLU) | (26, 26, 32) | 320 |
        | 2 | MaxPooling2D (2×2) | (13, 13, 32) | 0 |
        | 3 | Flatten | (5,408) | 0 |
        | 4 | Dense (128, ReLU) | (128) | 692,352 |
        | 5 | Dense (10, Softmax) | (10) | 1,290 |
        
        **Total Parameters: 693,962** (all trainable)
        
        **Training Configuration:**
        - Optimizer: Adam
        - Loss: Sparse Categorical Crossentropy
        - Metrics: Accuracy
        - Epochs: 5
        - Batch Size: 32 (default)
        """)

    # Deployment Architecture
    with st.expander("☁️ Deployment Architecture", expanded=False):
        st.markdown("""
        **Streamlit Cloud Ready**
        - Single-command deployment: `streamlit run app.py`
        - Model cached with `@st.cache_resource` (loaded once per session)
        - Data cached with `@st.cache_data` (preprocessing & MNIST loading)
        - No external dependencies beyond `requirements.txt`
        - Python 3.10+ compatible
        - CPU-only TensorFlow for cost efficiency
        - Relative paths only (deployment agnostic)
        """)

    # Navigation hint
    st.markdown("---")
    st.info("👈 **Use the sidebar navigation** to explore all features: Digit Recognition, Draw Digit, Image Upload, Dataset Explorer, Model Performance, Model Insights, and About.")

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Built with ❤️ by {DEVELOPER_INFO['name']} | "
        f"Powered by Streamlit & TensorFlow/Keras"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()