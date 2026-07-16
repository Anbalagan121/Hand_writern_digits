import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_training_history, load_test_metrics

def main():
    st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
    st.title("📈 Model Performance")
    st.markdown("Review the Convolutional Neural Network's performance during training and on unseen test data.")

    history = load_training_history()
    metrics = load_test_metrics()

    if not history or not metrics:
        st.warning("Training history or metrics not found. Please train the model first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Test Data Evaluation")
        test_acc = metrics.get("test_accuracy", 0)
        test_loss = metrics.get("test_loss", 0)
        
        c1, c2 = st.columns(2)
        c1.metric(label="Testing Accuracy", value=f"{test_acc * 100:.2f}%")
        c2.metric(label="Testing Loss", value=f"{test_loss:.4f}")
        
        st.info("The model achieves state-of-the-art accuracy on the MNIST dataset test split, indicating it generalizes well to unseen handwritten digits.")

    with col2:
        st.subheader("Training vs Validation Accuracy")
        epochs = list(range(1, len(history['accuracy']) + 1))
        df_acc = pd.DataFrame({
            'Epoch': epochs,
            'Train Accuracy': history['accuracy'],
            'Validation Accuracy': history['val_accuracy']
        })
        
        fig_acc = px.line(
            df_acc, 
            x='Epoch', 
            y=['Train Accuracy', 'Validation Accuracy'],
            title='Model Accuracy over Epochs',
            markers=True
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Training vs Validation Loss")
        df_loss = pd.DataFrame({
            'Epoch': epochs,
            'Train Loss': history['loss'],
            'Validation Loss': history['val_loss']
        })
        
        fig_loss = px.line(
            df_loss, 
            x='Epoch', 
            y=['Train Loss', 'Validation Loss'],
            title='Model Loss over Epochs',
            markers=True
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col4:
        st.subheader("Confusion Matrix")
        st.write("Visualizing model predictions against true labels on the 10,000 test images.")
        
        cm = metrics.get("confusion_matrix", [])
        if cm:
            x = [str(i) for i in range(10)]
            y = [str(i) for i in range(10)]
            
            # Using Plotly figure factory for annotated heatmap
            fig_cm = ff.create_annotated_heatmap(
                z=cm, 
                x=x, 
                y=y, 
                colorscale='Greens',
                showscale=True
            )
            fig_cm.update_layout(
                title_text='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label'
            )
            st.plotly_chart(fig_cm, use_container_width=True)

if __name__ == "__main__":
    main()
