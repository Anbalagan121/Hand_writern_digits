"""Model Insights page - Architecture, pipeline, and explanations."""

import streamlit as st
from components.ui_components import (
    page_header, divider, code_block, expander_section, two_column_layout
)
from services.model_service import load_model_cached, get_model_info, get_model_summary
from utils.exceptions import ModelLoadError


def main():
    page_header(
        "Model Insights",
        "Deep dive into the CNN architecture, prediction pipeline, and model understanding",
        "🧠"
    )
    
    # Load model for dynamic info
    try:
        model = load_model_cached()
        info = get_model_info(model)
    except ModelLoadError as e:
        st.error(str(e))
        model = None
        info = {
            "name": "sequential",
            "layers": 5,
            "total_params": 693962,
            "trainable_params": 693962,
            "non_trainable_params": 0,
            "input_shape": (None, 28, 28, 1),
            "output_shape": (None, 10),
        }
    
    # Architecture Summary
    st.markdown("### 🏗️ CNN Architecture Summary")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Model Configuration**")
        arch_data = {
            "Model Name": info["name"],
            "Total Layers": info["layers"],
            "Total Parameters": f"{info['total_params']:,}",
            "Trainable Parameters": f"{info['trainable_params']:,}",
            "Non-trainable": info["non_trainable_params"],
            "Input Shape": str(info["input_shape"]),
            "Output Shape": str(info["output_shape"]),
        }
        
        for key, val in arch_data.items():
            st.metric(key, val)
    
    with col2:
        st.markdown("**Layer Details**")
        st.code("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
                                                                 
 flatten (Flatten)           (None, 5408)              0         
                                                                 
 dense (Dense)               (None, 128)               692352    
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 693,962
Trainable params: 693,962
Non-trainable params: 0
_________________________________________________________________
        """, language="text")
    
    divider()
    
    # Layer-by-layer explanation
    st.markdown("### 🔍 Layer-by-Layer Breakdown")
    
    layers_info = [
        {
            "name": "Input Layer",
            "shape": "(28, 28, 1)",
            "desc": "Grayscale image, normalized to [0,1]. White digit on black background (MNIST format).",
            "params": 0
        },
        {
            "name": "Conv2D (32 filters, 3×3, ReLU)",
            "shape": "(26, 26, 32)",
            "desc": "32 learnable 3×3 kernels detect edges, corners, curves. ReLU adds non-linearity. 320 params (3×3×1×32 + 32 bias).",
            "params": 320
        },
        {
            "name": "MaxPooling2D (2×2)",
            "shape": "(13, 13, 32)",
            "desc": "Downsample by taking max in 2×2 windows. Reduces spatial dims, increases translation invariance. No parameters.",
            "params": 0
        },
        {
            "name": "Flatten",
            "shape": "(5,408)",
            "desc": "Reshape 13×13×32 = 5,408 features into 1D vector for dense layers. No parameters.",
            "params": 0
        },
        {
            "name": "Dense (128, ReLU)",
            "shape": "(128)",
            "desc": "Fully connected layer learns complex feature combinations. 5,408×128 + 128 = 692,352 params.",
            "params": 692352
        },
        {
            "name": "Dense (10, Softmax)",
            "shape": "(10)",
            "desc": "Output layer with 10 neurons (one per digit). Softmax converts logits to probabilities summing to 1. 128×10 + 10 = 1,290 params.",
            "params": 1290
        },
    ]
    
    for i, layer in enumerate(layers_info):
        with st.expander(f"Layer {i}: {layer['name']} — {layer['shape']} ({layer['params']:,} params)", expanded=i < 2):
            st.markdown(layer["desc"])
    
    divider()
    
    # Prediction Pipeline
    st.markdown("### ⚡ Prediction Pipeline")
    
    pipeline_steps = [
        ("1. Input", "User uploads/draws image (any size, color)"),
        ("2. Grayscale", "Convert to single channel (L mode)"),
        ("3. Invert", "MNIST format: white digit on black background"),
        ("4. Resize", "LANCZOS resampling to 28×28 pixels"),
        ("5. Normalize", "Divide by 255 → float32 in [0, 1]"),
        ("6. Reshape", "Add batch & channel dims → (1, 28, 28, 1)"),
        ("7. Forward Pass", "CNN inference: Conv → Pool → Flatten → Dense → Dense"),
        ("8. Softmax", "Logits → Probabilities (sum to 1.0)"),
        ("9. Argmax", "Highest probability = predicted digit"),
        ("10. Output", "Return digit, confidence, top-3 probabilities"),
    ]
    
    for step, desc in pipeline_steps:
        st.markdown(f"**{step}**: {desc}")
    
    divider()
    
    # How the model recognizes digits
    st.markdown("### 🎯 How the Model Recognizes Digits")
    
    st.markdown("""
    The CNN learns a **hierarchy of visual features**:
    
    1. **Conv2D Layer** (32 filters): 
       - Low-level features: vertical/horizontal edges, corners, small curves
       - Each filter specializes in detecting specific patterns
       - Example: Filter 1 detects vertical strokes (1, 7), Filter 2 detects loops (0, 6, 8, 9)
    
    2. **MaxPooling**: 
       - Creates translation invariance
       - "A curve here" ≈ "A curve there" 
       - Reduces computational cost
    
    3. **Dense Layer (128)**:
       - Combines features into digit concepts
       - Learns: "vertical stroke + loop at bottom = 9"
       - Non-linear combinations via ReLU
    
    4. **Output Layer (10)**:
       - Each neuron = one digit class
       - Softmax gives calibrated probabilities
       - Highest probability wins
    """)
    
    divider()
    
    # Advantages & Limitations
    col_adv, col_lim = st.columns(2)
    
    with col_adv:
        st.markdown("### ✅ Advantages")
        st.markdown("""
        - **High Accuracy**: 98.20% on MNIST test set
        - **Fast Inference**: < 50ms on CPU
        - **Translation Invariant**: MaxPool handles position shifts
        - **Parameter Efficient**: Only ~694K parameters
        - **No Feature Engineering**: Learns features automatically
        - **Robust to Noise**: Generalizes well to variations
        """)
    
    with col_lim:
        st.markdown("### ⚠️ Limitations")
        st.markdown("""
        - **MNIST Only**: Trained on clean, centered digits
        - **Single Digit**: Cannot handle multi-digit numbers
        - **Fixed Input**: Requires 28×28 preprocessing
        - **No Uncertainty**: Softmax can be overconfident
        - **Style Sensitivity**: May struggle with unusual handwriting
        - **No Spatial Reasoning**: Doesn't understand digit structure
        """)
    
    divider()
    
    # Future Improvements
    st.markdown("### 🚀 Future Improvements")
    
    improvements = [
        ("Architecture", [
            "Deeper CNN (VGG-style, ResNet)",
            "Batch Normalization for faster convergence",
            "Dropout for regularization",
            "Data Augmentation (rotation, shift, zoom)"
        ]),
        ("Training", [
            "More epochs with early stopping",
            "Learning rate scheduling",
            "Ensemble of multiple models",
            "Hyperparameter optimization (Optuna)"
        ]),
        ("Features", [
            "Multi-digit recognition (object detection)",
            "Real-time webcam inference",
            "Confidence calibration (temperature scaling)",
            "SHAP/LIME explainability integration"
        ]),
        ("Deployment", [
            "ONNX/TensorRT optimization",
            "Docker containerization",
            "API endpoint (FastAPI)",
            "Model versioning with MLflow"
        ]),
    ]
    
    for category, items in improvements:
        with expander_section(f"📦 {category}", expanded=False):
            for item in items:
                st.markdown(f"- {item}")
    
    # Model Summary
    divider()
    with expander_section("📄 Full Model Summary (from Keras)", expanded=False):
        if model:
            st.code(get_model_summary(model), language="text")
        else:
            st.warning("Model not loaded")


if __name__ == "__main__":
    main()