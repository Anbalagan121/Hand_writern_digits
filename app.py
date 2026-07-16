import streamlit as st

def main():
    st.set_page_config(
        page_title="Handwritten Digit Recognition",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏠 Handwritten Digit Recognition Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Project Overview")
        st.write("""
        Welcome to the **Handwritten Digit Recognition** AI Web Application!
        This premium dashboard showcases a complete end-to-end Deep Learning workflow for recognizing handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.
        """)

        st.subheader("Business Problem")
        st.info("""
        Accurate recognition of handwritten digits is crucial in various automated systems such as:
        - **Postal mail sorting**: Reading zip codes on envelopes.
        - **Bank check processing**: Automating the reading of check amounts.
        - **Form data entry**: Digitizing handwritten forms automatically.
        """)

        st.subheader("Key Features")
        st.success("""
        - ✍️ **Digit Prediction**: Upload an image and get predictions instantly.
        - 🎨 **Interactive Canvas**: Draw digits directly on the screen and see real-time AI inference.
        - 📊 **Dataset Explorer**: Explore the MNIST dataset distribution and samples.
        - 📈 **Model Performance**: Interactive visualizations of training metrics, accuracy, and confusion matrix.
        - 🧠 **Model Insights**: Dive deep into the CNN architecture and understanding.
        """)

    with col2:
        st.header("Technology Stack")
        st.markdown("""
        - **Frontend**: Streamlit
        - **Deep Learning**: TensorFlow & Keras
        - **Data Processing**: NumPy, Pandas, Pillow
        - **Visualizations**: Plotly, Seaborn, Matplotlib
        - **Machine Learning Ops**: Scikit-Learn
        """)
        
        st.header("Objectives")
        st.markdown("""
        1. **High Accuracy**: Achieve state-of-the-art accuracy using CNN.
        2. **Real-time Inference**: Provide instant predictions for user inputs.
        3. **Explainability**: Understand the model's performance and architecture.
        """)

    st.markdown("---")
    st.markdown("### Deep Learning Workflow")
    st.write("""
    1. **Data Collection**: Load the MNIST dataset.
    2. **Preprocessing**: Normalize, reshape, and clean the data.
    3. **Model Building**: Construct a deep Convolutional Neural Network.
    4. **Training**: Train the model on 60,000 training images.
    5. **Evaluation**: Test against 10,000 unseen images to validate accuracy.
    6. **Deployment**: Package into this interactive Streamlit application.
    """)

if __name__ == "__main__":
    main()
