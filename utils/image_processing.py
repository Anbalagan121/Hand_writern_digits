import numpy as np
from PIL import Image, ImageOps

def preprocess_uploaded_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an uploaded image (PNG/JPG/JPEG) to match MNIST training format.
    MNIST format: 28x28 pixels, grayscale, white digit on black background, normalized 0-1.
    """
    # Convert to grayscale
    img_gray = image.convert('L')
    
    # Invert the image (so digit is white, background is black)
    # Assuming user uploads black digit on white background
    img_inv = ImageOps.invert(img_gray)
    
    # Resize to 28x28
    img_resized = img_inv.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img_resized)
    
    # Normalize to 0-1
    img_norm = img_array / 255.0
    
    # Reshape for CNN input: (batch_size, height, width, channels)
    img_final = img_norm.reshape(1, 28, 28, 1)
    
    return img_final

def preprocess_canvas_image(image_data: np.ndarray) -> np.ndarray:
    """
    Preprocess image drawn on streamlit-drawable-canvas.
    The canvas usually returns a RGBA numpy array.
    """
    # Convert RGBA to grayscale Image
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    
    # Create a white background image
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3]) # paste using alpha channel as mask
    
    # Convert to grayscale
    img_gray = background.convert('L')
    
    # Invert to match MNIST (white on black)
    img_inv = ImageOps.invert(img_gray)
    
    # Resize to 28x28
    img_resized = img_inv.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img_resized)
    img_norm = img_array / 255.0
    
    # Reshape for CNN input
    img_final = img_norm.reshape(1, 28, 28, 1)
    
    return img_final
