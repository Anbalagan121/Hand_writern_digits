"""About page - Developer information and project details."""

import streamlit as st
from config.settings import DEVELOPER_INFO
from components.ui_components import page_header, divider, expander_section


def main():
    page_header(
        "About",
        "Learn about the developer and this project",
        "👨‍💻"
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
            width=200
        )
        st.markdown(f"### {DEVELOPER_INFO['name']}")
        st.markdown(f"**{DEVELOPER_INFO['role']}**")
        st.markdown(f"[🔗 GitHub Profile]({DEVELOPER_INFO['github']})")
    
    with col2:
        st.markdown("### Professional Roles")
        st.write("Principal AI Engineer | Senior Deep Learning Engineer | Computer Vision Engineer | Streamlit Architect | Software Architect")
        
        st.markdown("### Skills & Expertise")
        skills_df = st.dataframe(
            {"Category": ["Deep Learning", "Computer Vision", "Machine Learning", "Python", "TensorFlow/Keras", "PyTorch", "Streamlit", "MLOps", "Data Science"],
             "Proficiency": ["Expert", "Expert", "Expert", "Expert", "Expert", "Advanced", "Expert", "Advanced", "Expert"]},
            use_container_width=True, hide_index=True
        )
    
    divider()
    
    # Project Description
    st.markdown("### 📖 Project Description")
    st.markdown(DEVELOPER_INFO["project_description"])
    
    divider()
    
    # Technologies
    st.markdown("### 🛠️ Technologies Used")
    
    tech_categories = {
        "Core": ["Python 3.10+", "Streamlit", "TensorFlow/Keras"],
        "Data Science": ["NumPy", "Pandas", "Scikit-learn"],
        "Visualization": ["Plotly", "Matplotlib", "Seaborn"],
        "Image Processing": ["Pillow (PIL)"],
        "Development": ["Git", "VS Code", "Jupyter"],
    }
    
    for category, techs in tech_categories.items():
        with st.expander(f"{category} ({len(techs)})", expanded=False):
            for tech in techs:
                st.markdown(f"- {tech}")
    
    divider()
    
    # Deep Learning Tools
    st.markdown("### 🧠 Deep Learning Toolkit")
    for tool in DEVELOPER_INFO["deep_learning_tools"]:
        st.markdown(f"- {tool}")
    
    divider()
    
    # Future Enhancements
    st.markdown("### 🔮 Future Enhancements")
    for i, enhancement in enumerate(DEVELOPER_INFO["future_enhancements"], 1):
        st.markdown(f"{i}. {enhancement}")
    
    divider()
    
    # Architecture & Code Quality
    with expander_section("🏗️ Architecture & Code Quality", expanded=False):
        st.markdown("""
        **Modular Architecture:**
        - `app.py` - Main entry point with routing
        - `pages/` - 8 feature pages (Home, Digit Recognition, Draw Digit, Upload, Dataset, Performance, Insights, About)
        - `services/` - Business logic (model, data, image, prediction)
        - `components/` - Reusable UI (charts, cards, layouts)
        - `utils/` - Validation, exceptions, logging, caching
        - `config/` - Centralized settings
        
        **Production Practices:**
        - `@st.cache_resource` for model loading (once per session)
        - `@st.cache_data` for data & preprocessing
        - Type hints throughout
        - Custom exception hierarchy
        - Comprehensive logging
        - Input validation with clear error messages
        - Relative paths only (deployment ready)
        - PEP 8 compliant code style
        """)
    
    with expander_section("📊 Model Performance Summary", expanded=False):
        st.markdown("""
        **Test Set Results (10,000 images):**
        - Accuracy: **98.20%**
        - Loss: **0.0601**
        - Macro F1: **0.981**
        - Per-class F1: All > 0.97
        
        **Training (5 epochs):**
        - Final Train Acc: 99.57%
        - Final Val Acc: 98.20%
        - No overfitting observed
        - Converged smoothly
        
        **Comparison (from notebook):**
        | Model | Accuracy |
        |-------|----------|
        | Logistic Regression | 92.57% |
        | KNN (k=3) | 97.05% |
        | Random Forest | 97.04% |
        | SVM | 97.92% |
        | **CNN (This App)** | **98.59%** |
        """)
    
    with expander_section("📁 Project Structure", expanded=False):
        st.code("""
HandwrittenDigits/
├── app.py                      # Main entry point
├── train_and_save.py           # Training script
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── .gitignore                  # Git ignore rules
├── runtime.txt                 # Python version for deployment
├── config/
│   └── settings.py            # All configuration
├── services/
│   ├── model_service.py       # Model loading
│   ├── data_service.py        # Data loading
│   ├── image_service.py       # Image preprocessing
│   └── prediction_service.py  # Inference
├── components/
│   ├── ui_components.py       # Reusable UI
│   └── charts.py              # Plotly charts
├── utils/
│   ├── exceptions.py          # Custom exceptions
│   ├── validation.py          # Input validation
│   └── logging_config.py      # Logging setup
├── pages/
│   ├── 1_Home.py
│   ├── 2_Digit_Prediction.py
│   ├── 3_Draw_Digit.py
│   ├── 4_Image_Upload.py
│   ├── 5_Dataset_Explorer.py
│   ├── 6_Model_Performance.py
│   ├── 7_Model_Insights.py
│   └── 8_About.py
├── models/
│   └── cnn_model.keras        # Trained model (8.4 MB)
├── data/
│   ├── history.json           # Training history
│   └── metrics.json           # Test metrics
└── assets/                    # Static assets
        """, language="text")
    
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