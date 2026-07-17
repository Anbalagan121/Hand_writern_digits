"""Model Performance page - Training metrics and evaluation."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from components.ui_components import page_header, divider, metric_card, expander_section
from services.data_service import load_training_history, load_test_metrics
from utils.exceptions import DataLoadError


def main():
    page_header(
        "Model Performance",
        "Comprehensive evaluation of the CNN model on MNIST dataset",
        "📈"
    )
    
    # Load metrics
    try:
        history = load_training_history()
        metrics = load_test_metrics()
    except DataLoadError as e:
        st.error(str(e))
        st.stop()
    
    if not history or not metrics:
        st.warning("⚠️ Training history or metrics not found. Please run train_and_save.py first.")
        st.stop()
    
    # Test Set Performance
    st.markdown("### 🎯 Test Set Evaluation (10,000 images)")
    test_acc = metrics.get("test_accuracy", 0)
    test_loss = metrics.get("test_loss", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Test Accuracy", f"{test_acc * 100:.2f}%")
    with col2:
        metric_card("Test Loss", f"{test_loss:.4f}")
    with col3:
        metric_card("Error Rate", f"{(1 - test_acc) * 100:.2f}%")
    with col4:
        metric_card("Correct / Total", f"{int(test_acc * 10000):,} / 10,000")
    
    st.info(f"The model achieves **{test_acc * 100:.2f}% accuracy** on 10,000 unseen test images, "
            "demonstrating excellent generalization.")
    
    divider()
    
    # Training Curves
    st.markdown("### 📊 Training History")
    
    epochs = list(range(1, len(history["accuracy"]) + 1))
    
    tab1, tab2 = st.tabs(["📈 Accuracy", "📉 Loss"])
    
    with tab1:
        df_acc = pd.DataFrame({
            "Epoch": epochs,
            "Train": history["accuracy"],
            "Validation": history["val_accuracy"],
        })
        
        fig = px.line(
            df_acc, x="Epoch", y=["Train", "Validation"],
            title="Model Accuracy Over Epochs",
            markers=True, labels={"value": "Accuracy", "variable": "Set"}
        )
        fig.update_layout(yaxis=dict(range=[0, 1.05]), height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        with expander_section("📋 Accuracy Values", expanded=False):
            display_df = df_acc.copy()
            display_df["Train"] = (display_df["Train"] * 100).round(2).astype(str) + "%"
            display_df["Validation"] = (display_df["Validation"] * 100).round(2).astype(str) + "%"
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab2:
        df_loss = pd.DataFrame({
            "Epoch": epochs,
            "Train": history["loss"],
            "Validation": history["val_loss"],
        })
        
        fig = px.line(
            df_loss, x="Epoch", y=["Train", "Validation"],
            title="Model Loss Over Epochs",
            markers=True, labels={"value": "Loss", "variable": "Set"}
        )
        fig.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        with expander_section("📋 Loss Values", expanded=False):
            display_df = df_loss.copy()
            display_df["Train"] = display_df["Train"].round(4)
            display_df["Validation"] = display_df["Validation"].round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    divider()
    
    # Confusion Matrix
    st.markdown("### 🔍 Confusion Matrix (Test Set)")
    st.write("Predictions vs True labels on 10,000 test images.")
    
    cm = metrics.get("confusion_matrix", [])
    if cm:
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=[str(i) for i in range(10)],
            y=[str(i) for i in range(10)],
            colorscale="Greens",
            showscale=True,
            annotation_text=[[str(val) for val in row] for row in cm],
        )
        fig_cm.update_layout(
            title_text="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            xaxis=dict(side="bottom"),
            height=500,
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Per-class metrics
        st.markdown("#### Per-Class Performance")
        cm_array = np.array(cm)
        precision = cm_array.diagonal() / cm_array.sum(axis=0)
        recall = cm_array.diagonal() / cm_array.sum(axis=1)
        f1 = 2 * precision * recall / (precision + recall)
        
        class_df = pd.DataFrame({
            "Digit": range(10),
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": cm_array.sum(axis=1).astype(int),
        })
        
        fig_class = px.bar(
            class_df, x="Digit", y=["Precision", "Recall", "F1-Score"],
            title="Per-Class Metrics", barmode="group"
        )
        fig_class.update_layout(yaxis=dict(range=[0.9, 1.01]), height=350)
        st.plotly_chart(fig_class, use_container_width=True)
        
        st.dataframe(
            class_df.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
            use_container_width=True, hide_index=True
        )
    
    divider()
    
    # Model Comparison
    st.markdown("### 🏆 Model Comparison (from Training Notebook)")
    st.write("Comparison of different ML algorithms on MNIST from the exploration phase.")
    
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN (k=3)", "Random Forest", "SVM", "CNN (This App)"],
        "Accuracy": [0.9257, 0.9705, 0.9704, 0.9792, 0.9859],
    })
    
    fig_comp = px.bar(
        comparison_df, x="Model", y="Accuracy", text="Accuracy",
        title="Model Accuracy Comparison", color="Model",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_comp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_comp.update_layout(yaxis=dict(range=[0, 1.05]), xaxis_tickangle=-30, height=400)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    with expander_section("📝 Training Details", expanded=False):
        st.markdown("""
        **Training Configuration:**
        - Optimizer: Adam (default lr=0.001)
        - Loss: Sparse Categorical Crossentropy
        - Metrics: Accuracy
        - Epochs: 5
        - Batch Size: 32 (default)
        - Validation Split: Test set used as validation
        
        **Data Preprocessing:**
        - Reshape: (60000, 28, 28) → (60000, 28, 28, 1)
        - Normalize: / 255.0 → [0, 1] range
        - No data augmentation used
        
        **Hardware:** CPU training (TensorFlow CPU)
        """)


if __name__ == "__main__":
    main()