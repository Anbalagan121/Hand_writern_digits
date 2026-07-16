import streamlit as st

def main():
    st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
    st.title("🧠 Model Insights")
    st.markdown("Understand the architecture and inner workings of the Convolutional Neural Network (CNN).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("CNN Architecture Summary")
        st.write("The model used for digit recognition is a Sequential Convolutional Neural Network built with Keras.")
        
        st.code("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D)(None, 13, 13, 32)        0         
                                                                 
 flatten (Flatten)           (None, 5408)              0         
                                                                 
 dense (Dense)               (None, 128)               692352    
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 693,962
Trainable params: 693,962
Non-trainable params: 0
_________________________________________________________________
        """, language="text")

    with col2:
        st.subheader("How the Model Recognizes Digits")
        st.markdown("""
        1. **Input Layer (28x28x1)**: The model takes in a 28x28 pixel grayscale image.
        2. **Convolutional Layer (Conv2D)**: Applies 32 filters (3x3) to the image to extract features like edges, curves, and corners. The ReLU activation function introduces non-linearity.
        3. **Pooling Layer (MaxPooling2D)**: Reduces the spatial dimensions (down to 13x13) to decrease computational load and extract dominant features.
        4. **Flatten Layer**: Converts the 2D matrix into a 1D vector (5408 elements) so it can be fed into a standard Neural Network.
        5. **Fully Connected Layer (Dense)**: A layer with 128 neurons (ReLU activation) that learns non-linear combinations of the extracted features.
        6. **Output Layer (Dense)**: The final layer with 10 neurons (one for each digit 0-9). The Softmax activation converts the outputs into probability scores summing to 1.
        """)

        st.info("The prediction is simply the class (digit) with the highest probability score.")

if __name__ == "__main__":
    main()
