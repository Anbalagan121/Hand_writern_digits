"""Digit Recognition page - Upload image and get prediction."""

import streamlit as st
import time
from PIL import Image
from services.model_service import load_model_cached
from services.image_service import preprocess_uploaded_image, get_image_info
from services.prediction_service import predict_digit
from utils.exceptions import ModelLoadError, InvalidImageError, PredictionError
from utils.validation import validate_image_file, validate_image_content


st.set_page_config(page_title="Digit Recognition", page_icon="✍️", layout="wide")

st.title("✍️ Digit Recognition")
st.markdown("Upload a handwritten digit image (0-9) for AI prediction")
st.markdown("---")

# Load model
try:
    model = load_model_cached()
except ModelLoadError as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a single handwritten digit. Best results with black digit on white background."
    )

    if uploaded_file is not None:
        try:
            validate_image_file(uploaded_file)
            image = Image.open(uploaded_file)
            validate_image_content(image)

            st.image(image, caption="Uploaded Image", use_container_width=True)

            props = get_image_info(image)
            st.markdown("**Image Properties:**")
            cols = st.columns(len(props))
            for i, (key, value) in enumerate(props.items()):
                with cols[i]:
                    st.metric(key.capitalize(), value)

        except InvalidImageError as e:
            st.error(f"Invalid image: {e}")
            uploaded_file = None

with col2:
    st.subheader("🎯 Prediction Results")

    if uploaded_file is not None:
        with st.spinner("Processing image and predicting..."):
            try:
                start_time = time.time()

                processed_img, preview = preprocess_uploaded_image(image)
                result = predict_digit(model, processed_img)

                elapsed = (time.time() - start_time) * 1000

                # Preprocessed image
                st.markdown("**Preprocessed (28×28 MNIST format):**")
                st.image(preview, caption="Ready for model", width=120)

                # Results
                st.success(f"### Predicted Digit: {result['predicted_digit']}")
                st.metric("Confidence Score", f"{result['confidence']:.2f}%")
                st.caption(f"⏱️ Inference time: {elapsed:.1f} ms")

                st.markdown("#### Top 3 Predictions")
                for pred in result["top_3"]:
                    st.progress(
                        float(pred["probability"]),
                        text=f"Digit {pred['digit']}: {pred['percentage']:.2f}%"
                    )

                st.toast("Prediction complete!")

            except (Exception, PredictionError) as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("👆 Upload an image to see predictions here")