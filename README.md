# Handwritten Digit Recognition - Production Streamlit App

A production-ready AI web application for recognizing handwritten digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset. Built with professional software engineering practices.

## 🎯 Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, architecture, tech stack, workflow |
| ✍️ **Digit Recognition** | Upload image → instant prediction with confidence |
| 🖼️ **Upload Image** | Detailed analysis: preprocessing steps, probability distribution |
| 🎨 **Draw Digit** | Interactive canvas - draw & get real-time predictions |
| 📊 **Dataset Explorer** | MNIST statistics, class distribution, pixel analysis |
| 📈 **Model Performance** | Training curves, test metrics, confusion matrix |
| 🧠 **Model Insights** | CNN architecture, prediction pipeline, advantages |
| 👨‍💻 **About** | Developer info, skills, future enhancements |

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- 4GB+ RAM

### Installation
```bash
# Clone repository
git clone https://github.com/Anbalagan121/HandwrittenDigits.git
cd HandwrittenDigits

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
# or
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with `streamlit run app.py` or `streamlit run streamlit_app.py`

## 🏗️ Architecture

```
HandwrittenDigits/
├── app.py                 # Main entry point with routing
├── config/
│   └── settings.py        # Centralized configuration
├── components/
│   ├── ui_components.py   # Reusable UI components
│   └── charts.py          # Plotly chart builders
├── services/
│   ├── model_service.py   # Model loading & info
│   ├── data_service.py    # MNIST & metrics loading
│   ├── image_service.py   # Image preprocessing
│   └── prediction_service.py # Inference
├── utils/
│   ├── logging_config.py  # Logging setup
│   ├── exceptions.py      # Custom exceptions
│   └── validation.py      # Input validation
├── pages/
│   ├── 1_Home.py
│   ├── 2_Digit_Recognition.py
│   ├── 3_Draw_Digit.py
│   ├── 4_Image_Upload.py
│   ├── 5_Dataset_Explorer.py
│   ├── 6_Model_Performance.py
│   ├── 7_Model_Insights.py
│   └── 8_About.py
├── models/
│   └── cnn_model.keras    # Trained CNN (98.2% accuracy)
├── data/
│   ├── history.json       # Training history
│   └── metrics.json       # Test metrics & confusion matrix
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

## 🧠 Model Details

| Attribute | Value |
|-----------|-------|
| **Architecture** | CNN (Sequential) |
| **Layers** | Conv2D(32) → MaxPool2D → Flatten → Dense(128) → Dense(10) |
| **Parameters** | 693,962 (all trainable) |
| **Input** | 28×28×1 grayscale |
| **Output** | 10 classes (Softmax) |
| **Optimizer** | Adam |
| **Loss** | Sparse Categorical Crossentropy |
| **Epochs** | 5 |
| **Test Accuracy** | **98.20%** |
| **Test Loss** | 0.0601 |

## 🔧 Preprocessing Pipeline

The exact same preprocessing used during training:

```python
def preprocess_for_mnist(image: Image.Image) -> np.ndarray:
    # 1. Grayscale
    img = image.convert('L')
    
    # 2. Invert (MNIST: white digit on black)
    img = ImageOps.invert(img)
    
    # 3. Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 4. Normalize to [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0
    
    # 5. Add batch & channel dims
    return arr.reshape(1, 28, 28, 1)
```

## 📊 Screenshots

### Home Page
![Home](assets/home.png)

### Digit Recognition
![Recognition](assets/recognition.png)

### Draw Digit
![Draw](assets/draw.png)

### Model Performance
![Performance](assets/performance.png)

## 🛠️ Engineering Practices

- **Modular Architecture**: Separation of concerns (config, services, components, pages)
- **Caching**: `@st.cache_resource` for model, `@st.cache_data` for data
- **Error Handling**: Custom exceptions, graceful degradation
- **Input Validation**: File type, size, content validation
- **Logging**: Structured logging to file & console
- **Type Hints**: Full type annotations throughout
- **PEP 8**: Clean, consistent code style
- **Documentation**: Docstrings on all public functions

## 📈 Performance

- **Cold Start**: ~3-5 seconds (model loading)
- **Inference**: <100ms per prediction
- **Memory**: ~500MB (includes TensorFlow)
- **Concurrent Users**: Tested up to 10 simultaneous

## 🔮 Future Enhancements

- [ ] Custom dataset training support
- [ ] Webcam real-time recognition
- [ ] SHAP/LIME explainability
- [ ] Batch prediction API
- [ ] Docker containerization
- [ ] MLflow model versioning
- [ ] User authentication & history

## 👨‍💻 Developer

**Anbalagan K** - AI Engineer

- **GitHub**: [@Anbalagan121](https://github.com/Anbalagan121)
- **Roles**: Principal AI Engineer | Senior Deep Learning Engineer | Computer Vision Engineer | Streamlit Architect | Software Architect

## 📄 License

MIT License - Feel free to use for learning, portfolio, or commercial projects.

---

⭐ **Star this repo if you find it useful!**