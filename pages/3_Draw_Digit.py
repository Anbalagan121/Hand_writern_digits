"""Draw Digit page - Interactive canvas for drawing digits."""

import streamlit as st
import numpy as np
import time
from streamlit_drawable_canvas import st_canvas
from config.settings import (
    CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_STROKE_WIDTH,
    CANVAS_STROKE_COLOR, CANVAS_BG_COLOR, CANVAS_FILL_COLOR
)
from services.model_service import load_model_cached
from services.image_service import preprocess_canvas_image
from services.prediction_service import predict_digit
from utils.exceptions import ModelLoadError, PredictionError
from utils.validation import validate_canvas_data


st.set_page_config(page_title="Draw Digit", page_icon="🎨", layout="wide")

st.title("🎨 Draw a Digit")
st.markdown("Draw a single digit (0-9) on the canvas and get real-time AI prediction")
st.markdown("---")

# Load model
try:
    model = load_model_cached()
except ModelLoadError as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Drawing Canvas")
    st.caption("Draw a digit clearly. Use 'Clear' button to reset.")

    canvas_result = st_canvas(
        fill_color=CANVAS_FILL_COLOR,
        stroke_width=CANVAS_STROKE_WIDTH,
        stroke_color=CANVAS_STROKE_COLOR,
        background_color=CANVAS_BG_COLOR,
        update_streamlit=True,
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="freedraw",
        key="digit_canvas",
    )

    if st.button("🗑️ Clear Canvas", type="secondary", use_container_width=True):
        st.rerun()

with col2:
    st.subheader("🔮 Live Prediction")

    if canvas_result.image_data is not None:
        has_drawing = validate_canvas_data(canvas_result)

        if has_drawing:
            with st.spinner("Analyzing drawing..."):
                try:
                    start_time = time.time()

                    processed_img, preview = preprocess_canvas_image(canvas_result.image_data)
                    result = predict_digit(model, processed_img)

                    elapsed = (time.time() - start_time) * 1000

                    # Preprocessed image
                    st.markdown("**Preprocessed (28×28):**")
                    st.image(preview, caption="MNIST Format", width=100)

                    # Results
                    st.success(f"### Predicted Digit: {result['predicted_digit']}")
                    st.metric("Confidence", f"{result['confidence']:.2f}%")
                    st.caption(f"⏱️ {elapsed:.1f} ms")

                    st.markdown("#### Top 3")
                    for pred in result["top_3"]:
                        st.progress(
                            float(pred["probability"]),
                            text=f"Digit {pred['digit']}: {pred['percentage']:.2f}%"
                        )

                    st.toast("Prediction updated!")

                except (Exception, PredictionError) as e:
                    st.error(f"Error: {e}")
        else:
            st.info("👆 Draw a digit on the canvas to see prediction")
    else:
        st.info("👆 Draw a digit on the canvas to see prediction")