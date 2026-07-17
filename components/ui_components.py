"""Reusable UI components."""

from contextlib import contextmanager
from typing import Optional, List, Callable, Any

import streamlit as st


def page_header(title: str, subtitle: Optional[str] = None, icon: str = "") -> None:
    """Render page header with title and optional subtitle."""
    st.markdown(f"# {icon} {title}")
    if subtitle:
        st.markdown(f"### {subtitle}")
    st.markdown("---")


def info_card(title: str, content: str, icon: str = "ℹ️") -> None:
    """Render an info card."""
    with st.container():
        st.markdown(f"#### {icon} {title}")
        st.info(content)


def success_card(title: str, content: str, icon: str = "✅") -> None:
    """Render a success card."""
    with st.container():
        st.markdown(f"#### {icon} {title}")
        st.success(content)


def warning_card(title: str, content: str, icon: str = "⚠️") -> None:
    """Render a warning card."""
    with st.container():
        st.markdown(f"#### {icon} {title}")
        st.warning(content)


def error_card(title: str, content: str, icon: str = "❌") -> None:
    """Render an error card."""
    with st.container():
        st.markdown(f"#### {icon} {title}")
        st.error(content)


def metric_card(label: str, value: str, delta: Optional[str] = None, delta_color: str = "normal") -> None:
    """Render a metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def prediction_display(result: dict) -> None:
    """Render prediction results with confidence and top-3."""
    st.success(f"### Predicted Digit: {result['predicted_digit']}")
    st.metric("Confidence Score", f"{result['confidence']:.2f}%")
    
    st.markdown("#### Top 3 Predictions")
    for pred in result["top_3"]:
        st.progress(
            float(pred["probability"]),
            text=f"Digit {pred['digit']}: {pred['percentage']:.2f}%"
        )


def progress_bar_with_label(value: float, label: str) -> None:
    """Render progress bar with label."""
    st.progress(value, text=label)


def image_preview(image, caption: str = "", width: int = 150) -> None:
    """Display image preview."""
    st.image(image, caption=caption, width=width)


def image_properties_display(properties: dict) -> None:
    """Display image properties in columns."""
    cols = st.columns(len(properties))
    for i, (key, value) in enumerate(properties.items()):
        with cols[i]:
            st.metric(key.capitalize(), value)


def two_column_layout(
    left_content: Any, 
    right_content: Any, 
    ratio: List[int] = [1, 1],
    left_callable: bool = False,
    right_callable: bool = False
) -> None:
    """Render two-column layout."""
    col1, col2 = st.columns(ratio)
    with col1:
        if left_callable:
            left_content()
        else:
            st.markdown(left_content)
    with col2:
        if right_callable:
            right_content()
        else:
            st.markdown(right_content)


def three_column_layout(contents: List[Any], ratio: List[int] = [1, 1, 1]) -> None:
    """Render three-column layout."""
    cols = st.columns(ratio)
    for col, content in zip(cols, contents):
        with col:
            if callable(content):
                content()
            else:
                st.markdown(content)


def grid_layout(items: List[Any], cols: int = 5) -> None:
    """Render grid of items."""
    for i in range(0, len(items), cols):
        row_cols = st.columns(cols)
        for j, col in enumerate(row_cols):
            idx = i + j
            if idx < len(items):
                with col:
                    item = items[idx]
                    if callable(item):
                        item()
                    else:
                        st.markdown(item)


def divider() -> None:
    """Render horizontal divider."""
    st.markdown("---")


@contextmanager
def expander_section(title: str, content: Optional[str] = None, expanded: bool = False):
    """Render an expandable section and support both simple and content-driven usage."""
    with st.expander(title, expanded=expanded):
        if content is not None:
            st.markdown(content)
        yield


def tabs_section(tab_labels: List[str], tab_contents: List[Any]) -> None:
    """Render tabbed content."""
    tabs = st.tabs(tab_labels)
    for tab, content in zip(tabs, tab_contents):
        with tab:
            if callable(content):
                content()
            else:
                st.markdown(content)


def code_block(code: str, language: str = "python") -> None:
    """Render code block."""
    st.code(code, language=language)


def empty_state(message: str, icon: str = "📭") -> None:
    """Render empty state message."""
    st.markdown(f"### {icon} {message}")


def loading_spinner(message: str = "Processing..."):
    """Context manager for loading spinner."""
    return st.spinner(message)


def toast_message(message: str, icon: str = "✅") -> None:
    """Show toast notification."""
    st.toast(f"{icon} {message}")


def sidebar_navigation(pages: List[dict]) -> int:
    """Render sidebar navigation and return selected index."""
    st.sidebar.title("🧭 Navigation")
    
    page_names = [f"{p['icon']} {p['name']}" for p in pages]
    selected = st.sidebar.radio("Go to", range(len(page_names)), format_func=lambda x: page_names[x], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    return selected