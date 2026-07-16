import streamlit as st

def main():
    st.set_page_config(page_title="About the Developer", page_icon="📋", layout="wide")
    st.title("📋 About")
    st.markdown("Learn more about the developer behind this Deep Learning web application.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
            width=200
        )
        st.markdown("### Anbalagan K")
        st.markdown("**AI Engineer**")
        st.markdown("[🔗 GitHub Profile](https://github.com/Anbalagan121)")

    with col2:
        st.subheader("Professional Roles")
        st.write("Principal AI Engineer | Senior Deep Learning Engineer | Computer Vision Engineer | Streamlit Architect | Software Architect")
        
        st.subheader("Skills & Technologies")
        st.markdown("""
        - **Programming**: Python
        - **Deep Learning**: TensorFlow, Keras, PyTorch, CNNs, Neural Networks
        - **Machine Learning**: Scikit-Learn, Random Forests, SVM, KNN, Logistic Regression
        - **Web Apps**: Streamlit, Next.js, Vite
        - **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn, Plotly
        """)

        st.subheader("Future Enhancements")
        st.info("""
        - Implement transfer learning architectures (e.g., ResNet).
        - Expand to classify handwritten alphabets (EMNIST dataset).
        - Support uploading multiple images in bulk.
        - Add a darker, customizable theme option.
        """)

if __name__ == "__main__":
    main()
