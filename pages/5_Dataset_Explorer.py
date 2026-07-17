"""Dataset Explorer page - MNIST dataset visualization."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from components.ui_components import page_header, divider, expander_section
from services.data_service import load_mnist, get_class_distribution, get_sample_indices
from utils.exceptions import DataLoadError


def main():
    page_header(
        "Dataset Explorer",
        "Explore the MNIST handwritten digit dataset statistics and samples",
        "📊"
    )
    
    # Load MNIST data
    with st.spinner("Loading MNIST dataset..."):
        try:
            x_train, y_train, x_test, y_test = load_mnist()
        except DataLoadError as e:
            st.error(str(e))
            st.stop()
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Images", f"{x_train.shape[0]:,}")
    with col2:
        st.metric("Test Images", f"{x_test.shape[0]:,}")
    with col3:
        st.metric("Image Resolution", f"{x_train.shape[1]}×{x_train.shape[2]}")
    with col4:
        st.metric("Classes", "10 (0-9)")
    
    st.markdown("""
    The **MNIST** (Modified National Institute of Standards and Technology) database 
    contains 70,000 grayscale images of handwritten digits, split into 60,000 training 
    and 10,000 test images. Each image is 28×28 pixels with values 0-255.
    """)
    
    divider()
    
    # Class Distribution
    st.markdown("### 📊 Class Distribution (Training Set)")
    
    unique, counts = get_class_distribution(y_train)
    dist_df = pd.DataFrame({"Digit": unique, "Count": counts})
    
    col_dist1, col_dist2 = st.columns([2, 1])
    
    with col_dist1:
        fig = px.bar(
            dist_df, x="Digit", y="Count", text="Count",
            title="Digit Distribution in Training Data",
            color="Digit", color_continuous_scale="Viridis"
        )
        fig.update_layout(xaxis_type="category", showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_dist2:
        st.markdown("**Statistics:**")
        stats_df = dist_df.copy()
        stats_df["Percentage"] = (stats_df["Count"] / stats_df["Count"].sum() * 100).round(2)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.metric("Mean per class", f"{counts.mean():.0f}")
        st.metric("Std deviation", f"{counts.std():.0f}")
        st.metric("Min/Max", f"{counts.min()}/{counts.max()}")
    
    divider()
    
    # Sample Images
    st.markdown("### 🖼️ Sample Images (First of Each Class)")
    
    sample_indices = get_sample_indices(y_train)
    cols = st.columns(10)
    
    for i, idx in enumerate(sample_indices):
        with cols[i]:
            st.image(x_train[idx], caption=f"Label: {i}", use_container_width=True)
    
    # Extended sample grid
    with expander_section("🔍 More Samples (5 per class)", expanded=False):
        for digit in range(10):
            indices = np.where(y_train == digit)[0][:5]
            cols = st.columns(5)
            for j, idx in enumerate(indices):
                with cols[j]:
                    st.image(x_train[idx], caption=f"{digit}", use_container_width=True)
    
    divider()
    
    # Pixel Statistics
    st.markdown("### 📈 Pixel Intensity Statistics")
    
    # Overall statistics
    all_pixels_train = x_train.flatten()
    all_pixels_test = x_test.flatten()
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown("**Training Set Pixel Values:**")
        st.write(f"- Mean: {all_pixels_train.mean():.2f}")
        st.write(f"- Std: {all_pixels_train.std():.2f}")
        st.write(f"- Min: {all_pixels_train.min()}")
        st.write(f"- Max: {all_pixels_train.max()}")
        st.write(f"- Median: {np.median(all_pixels_train):.2f}")
    
    with col_stat2:
        st.markdown("**Test Set Pixel Values:**")
        st.write(f"- Mean: {all_pixels_test.mean():.2f}")
        st.write(f"- Std: {all_pixels_test.std():.2f}")
        st.write(f"- Min: {all_pixels_test.min()}")
        st.write(f"- Max: {all_pixels_test.max()}")
        st.write(f"- Median: {np.median(all_pixels_test):.2f}")
    
    # Pixel histogram
    fig_hist = px.histogram(
        x=all_pixels_train, nbins=50,
        title="Pixel Value Distribution (Training Set)",
        labels={"x": "Pixel Value (0=Black, 255=White)", "y": "Frequency"}
    )
    fig_hist.update_layout(bargap=0.05, height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Per-digit pixel statistics
    with expander_section("📊 Per-Digit Pixel Statistics", expanded=False):
        digit_stats = []
        for d in range(10):
            mask = y_train == d
            pixels = x_train[mask].flatten()
            digit_stats.append({
                "Digit": d,
                "Mean": f"{pixels.mean():.2f}",
                "Std": f"{pixels.std():.2f}",
                "Median": f"{np.median(pixels):.2f}",
                "Sparsity (% zeros)": f"{(pixels == 0).mean() * 100:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(digit_stats), use_container_width=True, hide_index=True)
    
    divider()
    
    # Dataset Info
    with expander_section("ℹ️ Dataset Information", expanded=False):
        st.markdown("""
        **MNIST Dataset Details:**
        - **Source**: NIST Special Database 19 (converted by Yann LeCun et al.)
        - **Total Images**: 70,000 (60,000 train + 10,000 test)
        - **Image Size**: 28 × 28 pixels
        - **Color Depth**: 8-bit grayscale (0-255)
        - **Classes**: 10 digits (0-9)
        - **Preprocessing**: Centered, size-normalized, anti-aliased
        
        **Train/Test Split:**
        - Training: 60,000 images (6,000 per class)
        - Test: 10,000 images (~1,000 per class)
        
        **Preprocessing for CNN:**
        - Reshape: (28, 28) → (28, 28, 1)
        - Normalize: Divide by 255.0 → [0, 1]
        - Invert: White digit on black background (MNIST format)
        """)


if __name__ == "__main__":
    main()