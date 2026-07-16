import streamlit as st
from PIL import Image
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.image_processing import preprocess_uploaded_image
from utils.model_loader import load_cnn_model

def main():
    st.set_page_config(page_title="Digit Prediction", page_icon="✍️", layout="wide")
    st.title("✍️ Digit Prediction")
    st.markdown("Upload a handwritten digit image (0-9) and let the CNN model predict it!")

    # Load model
    try:
        model = load_cnn_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a PNG, JPG or JPEG image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            st.info(f"**Image Properties:**\n- Format: {image.format}\n- Size: {image.size}\n- Mode: {image.mode}")

    with col2:
        if uploaded_file is not None:
            st.subheader("Prediction Results")
            with st.spinner("Processing image and predicting..."):
                try:
                    # Preprocess
                    processed_img = preprocess_uploaded_image(image)
                    
                    # Predict
                    predictions = model.predict(processed_img)[0]
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[predicted_class] * 100
                    
                    # Display preprocessed image (for debugging/visual insight)
                    st.image(processed_img[0, :, :, 0], caption="Preprocessed Image (28x28 Grayscale)", width=150)
                    
                    # Display results
                    st.success(f"### Predicted Digit: {predicted_class}")
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                    
                    # Top 3 predictions
                    st.markdown("#### Top 3 Predictions")
                    top_3_indices = np.argsort(predictions)[-3:][::-1]
                    
                    for idx in top_3_indices:
                        st.progress(float(predictions[idx]), text=f"Digit {idx}: {predictions[idx]*100:.2f}%")
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
