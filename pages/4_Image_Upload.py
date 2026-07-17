"""Image Upload page - Detailed image analysis and prediction."""

import streamlit as st
import time
import plotly.express as px
import pandas as pd
from PIL import Image, ImageOps
from services.model_service import load_model_cached
from services.image_service import preprocess_uploaded_image, get_image_info
from services.prediction_service import predict_digit
from utils.exceptions import ModelLoadError, InvalidImageError, PredictionError
from utils.validation import validate_image_file, validate_image_content


st.set_page_config(page_title="Upload Image", page_icon="🖼️", layout="wide")

st.title("🖼️ Upload Image")
st.markdown("Upload a handwritten digit for detailed analysis and prediction")
st.markdown("---")

# Load model
try:
    model = load_model_cached()
except ModelLoadError as e:
    st.error(str(e))
    st.stop()

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

        props = get_image_info(image)

        # Tabs for organized display
        tab1, tab2, tab3, tab4 = st.tabs([
            "📸 Original", "🔧 Preprocessing", "🎯 Prediction", "📋 Details"
        ])

        with tab1:
            st.markdown("### Original Image")
            st.image(image, caption=f"Original: {image.size[0]}×{image.size[1]}", use_container_width=True)
            st.markdown("**Properties:**")
            cols = st.columns(len(props))
            for i, (key, value) in enumerate(props.items()):
                with cols[i]:
                    st.metric(key.capitalize(), value)

        with tab2:
            st.markdown("### Preprocessing Pipeline")
            st.markdown("""
            The image goes through these steps to match MNIST training format:
            1. **Grayscale** → Single channel
            2. **Invert** → MNIST uses white digit on black background
            3. **Resize** → 28×28 pixels (LANCZOS)
            4. **Normalize** → Pixel values [0, 1]
            5. **Reshape** → (1, 28, 28, 1) for CNN input
            """)

            with st.spinner("Preprocessing..."):
                processed_img, preview = preprocess_uploaded_image(image)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Grayscale**")
                gray = image.convert("L")
                st.image(gray, use_container_width=True)

            with col_b:
                st.markdown("**Inverted**")
                inverted = ImageOps.invert(gray)
                st.image(inverted, use_container_width=True)

            st.markdown("**Final 28×28 (MNIST Format)**")
            st.image(preview, caption="Model Input", width=150)

        with tab3:
            st.markdown("### Prediction Results")

            with st.spinner("Running inference..."):
                start_time = time.time()
                result = predict_digit(model, processed_img)
                elapsed = (time.time() - start_time) * 1000

            st.success(f"### Predicted Digit: {result['predicted_digit']}")
            st.metric("Confidence", f"{result['confidence']:.2f}%")
            st.caption(f"⏱️ Inference time: {elapsed:.1f} ms")

            st.markdown("#### Top 3 Predictions")
            for pred in result["top_3"]:
                st.progress(
                    float(pred["probability"]),
                    text=f"Digit {pred['digit']}: {pred['percentage']:.2f}%"
                )

            # Probability distribution chart
            st.markdown("#### Probability Distribution")
            probs = result["all_probabilities"]
            df = pd.DataFrame({
                "Digit": list(range(10)),
                "Probability": probs
            })
            fig = px.bar(
                df, x="Digit", y="Probability",
                title="Class Probabilities",
                color="Probability", color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_type="category", height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.toast("Prediction complete!")

        with tab4:
            st.markdown("### Technical Details")

            st.markdown("#### Input Image")
            st.json({
                "filename": uploaded_file.name,
                "format": props["format"],
                "size": f"{props['width']}×{props['height']}",
                "mode": props["mode"],
                "file_size_kb": round(uploaded_file.size / 1024, 1)
            })

            st.markdown("#### Preprocessed Array")
            st.json({
                "shape": list(processed_img.shape),
                "dtype": str(processed_img.dtype),
                "min": float(processed_img.min()),
                "max": float(processed_img.max()),
                "mean": float(processed_img.mean())
            })

            st.markdown("#### Model Output")
            st.json({
                "predicted_class": result["predicted_digit"],
                "confidence": f"{result['confidence']:.2f}%",
                "top_3": [
                    {"digit": p["digit"], "probability": f"{p['percentage']:.2f}%"}
                    for p in result["top_3"]
                ]
            })

    except InvalidImageError as e:
        st.error(f"Invalid image: {e}")
else:
    st.info("👆 Upload an image to begin analysis")

    with st.expander("📝 Tips for Best Results", expanded=False):
        st.markdown("""
        **For optimal recognition accuracy:**
        - Use **black ink on white paper** (or dark on light)
        - Write a **single digit** clearly in the center
        - Avoid shadows, grid lines, or multiple digits
        - Ensure good lighting and contrast
        - Image will be auto-inverted (MNIST format: white on black)
        
        **Supported formats:** PNG, JPG, JPEG  
        **Max file size:** 10 MB
        """)