import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.datasets import mnist

@st.cache_data
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test

def main():
    st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")
    st.title("📊 Dataset Explorer")
    st.markdown("Explore the **MNIST** (Modified National Institute of Standards and Technology) database of handwritten digits.")

    with st.spinner("Loading MNIST dataset..."):
        x_train, y_train, x_test, y_test = load_mnist_data()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.write("The MNIST database contains 70,000 grayscale images of handwritten digits, divided into a training set and a test set.")
        
        st.metric(label="Training Images", value=f"{x_train.shape[0]:,}")
        st.metric(label="Testing Images", value=f"{x_test.shape[0]:,}")
        st.metric(label="Image Resolution", value=f"{x_train.shape[1]}x{x_train.shape[2]} pixels")
        st.metric(label="Number of Classes", value="10 (Digits 0-9)")

    with col2:
        st.subheader("Class Distribution (Training Set)")
        # Plot class distribution using Plotly
        unique, counts = np.unique(y_train, return_counts=True)
        dist_df = pd.DataFrame({'Digit': unique, 'Count': counts})
        
        fig = px.bar(
            dist_df, 
            x='Digit', 
            y='Count', 
            text='Count', 
            title='Distribution of Digits in Training Data',
            color='Digit',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(xaxis_type='category')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Digit Images")
    st.write("Below are some sample images from the dataset.")

    # Show a grid of images
    cols = st.columns(10)
    for i in range(10):
        with cols[i]:
            # Find the first occurrence of each digit
            idx = np.where(y_train == i)[0][0]
            st.image(x_train[idx], caption=f"Label: {i}", use_container_width=True)

    st.markdown("---")
    st.subheader("Pixel Distribution")
    st.write("Distribution of pixel intensities in a sample image.")
    sample_img = x_train[0]
    fig_hist = px.histogram(
        x=sample_img.flatten(), 
        nbins=20, 
        title="Pixel Value Histogram (0=Black, 255=White)",
        labels={'x': 'Pixel Value', 'y': 'Frequency'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()
