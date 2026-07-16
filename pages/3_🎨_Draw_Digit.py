import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.image_processing import preprocess_canvas_image
from utils.model_loader import load_cnn_model

def main():
    st.set_page_config(page_title="Draw Digit", page_icon="🎨", layout="wide")
    st.title("🎨 Draw a Digit")
    st.markdown("Use your mouse or touch screen to draw a single digit (0-9) in the box below.")

    try:
        model = load_cnn_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Interactive Canvas")
        st.write("Draw smoothly. If you make a mistake, clear and try again.")
        
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
            stroke_width=20,
            stroke_color="#000000",
            background_color="#FFFFFF",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        st.subheader("Live Prediction")
        
        if canvas_result.image_data is not None:
            # Check if canvas is empty by checking if there's any non-white/transparent pixel
            # The background is white. Any drawing is black. 
            # image_data is RGBA. Black is [0, 0, 0, 255], White is [255, 255, 255, 255]
            if np.any(canvas_result.image_data[:, :, 0:3] < 255):
                with st.spinner("Analyzing drawing..."):
                    try:
                        # Preprocess the canvas drawing
                        processed_img = preprocess_canvas_image(canvas_result.image_data)
                        
                        # Predict
                        predictions = model.predict(processed_img)[0]
                        predicted_class = np.argmax(predictions)
                        confidence = predictions[predicted_class] * 100
                        
                        st.image(processed_img[0, :, :, 0], caption="Preprocessed Image (28x28 Grayscale)", width=100)
                        
                        st.success(f"### Predicted Digit: {predicted_class}")
                        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                        
                        st.markdown("#### Top 3 Predictions")
                        top_3_indices = np.argsort(predictions)[-3:][::-1]
                        
                        for idx in top_3_indices:
                            st.progress(float(predictions[idx]), text=f"Digit {idx}: {predictions[idx]*100:.2f}%")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
            else:
                st.info("Start drawing on the canvas to see the prediction here.")

if __name__ == "__main__":
    main()
