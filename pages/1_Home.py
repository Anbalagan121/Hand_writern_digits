"""Home page - Project overview and introduction."""

import streamlit as st
from components.ui_components import (
    page_header, info_card, success_card, two_column_layout, 
    divider, expander_section, metric_card
)


def main():
    page_header(
        "Handwritten Digit Recognition",
        "Production-ready AI web application powered by CNN and Streamlit",
        "🏠"
    )
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Project Overview
        
        Welcome to the **Handwritten Digit Recognition** AI Web Application! This production-ready 
        dashboard showcases a complete end-to-end Deep Learning workflow for recognizing handwritten 
        digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
        
        The application demonstrates professional software engineering practices including modular 
        architecture, caching, error handling, logging, and responsive UI design.
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
        col_obj1, col_obj2, col_obj3 = st.columns(3)
        with col_obj1:
            metric_card("Accuracy", "98.20%", "Test Set", "normal")
        with col_obj2:
            metric_card("Inference", "< 100ms", "Real-time", "normal")
        with col_obj3:
            metric_card("Architecture", "CNN", "5 Layers", "normal")
    
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
            "🖼️ **Image Upload** - Support PNG, JPG, JPEG formats",
            "📊 **Dataset Explorer** - MNIST statistics and samples",
            "📈 **Model Performance** - Interactive charts and confusion matrix",
            "🧠 **Model Insights** - Architecture and prediction pipeline",
        ]
        for feature in features:
            st.markdown(feature)
    
    divider()
    
    # Deep Learning Workflow
    st.markdown("### 🔬 Deep Learning Workflow")
    workflow_steps = [
        ("1. Data Collection", "Load MNIST dataset (60K train + 10K test)"),
        ("2. Preprocessing", "Normalize, reshape to 28×28×1, invert colors"),
        ("3. Model Building", "Construct CNN with Conv2D, MaxPool, Dense layers"),
        ("4. Training", "Train 5 epochs with Adam optimizer"),
        ("5. Evaluation", "Validate on 10K unseen test images"),
        ("6. Deployment", "Package as Streamlit web application"),
    ]
    
    for i, (step, desc) in enumerate(workflow_steps):
        col_a, col_b = st.columns([1, 4])
        with col_a:
            st.markdown(f"**{step}**")
        with col_b:
            st.markdown(desc)
    
    divider()
    
    # Architecture Overview
    with st.expander("🏗️ CNN Architecture Details", expanded=False):
        st.markdown("""
        **Model: Sequential CNN**
        
        | Layer | Type | Output Shape | Parameters |
        |-------|------|--------------|------------|
        | 1 | Conv2D (32 filters, 3×3, ReLU) | (26, 26, 32) | 320 |
        | 2 | MaxPooling2D (2×2) | (13, 13, 32) | 0 |
        | 3 | Flatten | (5408,) | 0 |
        | 4 | Dense (128, ReLU) | (128,) | 692,352 |
        | 5 | Dense (10, Softmax) | (10,) | 1,290 |
        
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
        - Model cached with `@st.cache_resource`
        - Data cached with `@st.cache_data`
        - No external dependencies beyond requirements.txt
        - Python 3.10+ compatible
        - CPU-only TensorFlow for cost efficiency
        """)


if __name__ == "__main__":
    main()