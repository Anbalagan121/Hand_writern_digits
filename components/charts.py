"""Chart components using Plotly."""

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def class_distribution_bar(unique, counts, title: str = "Class Distribution") -> go.Figure:
    """Create class distribution bar chart."""
    df = pd.DataFrame({"Digit": unique, "Count": counts})
    
    fig = px.bar(
        df,
        x="Digit",
        y="Count",
        text="Count",
        title=title,
        color="Digit",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        xaxis_type="category",
        showlegend=False,
        xaxis_title="Digit",
        yaxis_title="Count",
        height=400,
    )
    return fig


def pixel_histogram(pixel_values, title: str = "Pixel Value Distribution") -> go.Figure:
    """Create pixel value histogram."""
    fig = px.histogram(
        x=pixel_values.flatten(),
        nbins=20,
        title=title,
        labels={"x": "Pixel Value (0=Black, 255=White)", "y": "Frequency"},
    )
    fig.update_layout(bargap=0.1, height=400)
    return fig


def training_accuracy_plot(history: dict) -> go.Figure:
    """Create training vs validation accuracy plot."""
    epochs = list(range(1, len(history["accuracy"]) + 1))
    df = pd.DataFrame({
        "Epoch": epochs,
        "Train Accuracy": history["accuracy"],
        "Validation Accuracy": history["val_accuracy"],
    })
    
    fig = px.line(
        df,
        x="Epoch",
        y=["Train Accuracy", "Validation Accuracy"],
        title="Model Accuracy Over Epochs",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        height=400,
    )
    return fig


def training_loss_plot(history: dict) -> go.Figure:
    """Create training vs validation loss plot."""
    epochs = list(range(1, len(history["loss"]) + 1))
    df = pd.DataFrame({
        "Epoch": epochs,
        "Train Loss": history["loss"],
        "Validation Loss": history["val_loss"],
    })
    
    fig = px.line(
        df,
        x="Epoch",
        y=["Train Loss", "Validation Loss"],
        title="Model Loss Over Epochs",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        height=400,
    )
    return fig


def confusion_matrix_heatmap(cm: list, title: str = "Confusion Matrix") -> go.Figure:
    """Create confusion matrix heatmap."""
    x_labels = [str(i) for i in range(10)]
    y_labels = [str(i) for i in range(10)]
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=x_labels,
        y=y_labels,
        colorscale="Greens",
        showscale=True,
        annotation_text=[[str(val) for val in row] for row in cm],
    )
    fig.update_layout(
        title_text=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(side="bottom"),
        height=500,
    )
    return fig


def model_comparison_bar(models: list, accuracies: list, title: str = "Model Comparison") -> go.Figure:
    """Create model comparison bar chart."""
    df = pd.DataFrame({"Model": models, "Accuracy": accuracies})
    
    fig = px.bar(
        df,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title=title,
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        yaxis=dict(range=[0, 1.05]),
        xaxis_tickangle=-30,
        showlegend=False,
        height=400,
    )
    return fig


def architecture_diagram() -> go.Figure:
    """Create CNN architecture flow diagram."""
    layers = [
        ("Input", "28×28×1"),
        ("Conv2D", "26×26×32"),
        ("MaxPool", "13×13×32"),
        ("Flatten", "5408"),
        ("Dense", "128"),
        ("Output", "10"),
    ]
    
    fig = go.Figure()
    y_pos = list(range(len(layers)))[::-1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for i, ((name, shape), y, color) in enumerate(zip(layers, y_pos, colors)):
        fig.add_shape(
            type="rect",
            x0=0, y0=y - 0.4, x1=1, y1=y + 0.4,
            fillcolor=color, opacity=0.7,
            line=dict(color="white", width=2),
        )
        fig.add_annotation(
            x=0.5, y=y,
            text=f"<b>{name}</b><br>{shape}",
            showarrow=False,
            font=dict(size=14, color="white"),
        )
    
    for i in range(len(layers) - 1):
        fig.add_annotation(
            x=0.5, y=y_pos[i] - 0.4,
            ax=0.5, ay=y_pos[i + 1] + 0.4,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=2, arrowcolor="gray",
        )
    
    fig.update_layout(
        title="CNN Architecture Flow",
        xaxis=dict(visible=False, range=[-0.5, 1.5]),
        yaxis=dict(visible=False, range=[-1, len(layers)]),
        height=450,
        showlegend=False,
    )
    return fig