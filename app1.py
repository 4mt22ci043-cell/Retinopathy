import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #e6f7ff, #ccffeb);
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2b6777;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #297373;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_retinopathy_model():
    model = load_model("retinopathy.h5")
    return model

model = load_retinopathy_model()

# Title
st.markdown("<div class='title'>🩺 Diabetic Retinopathy Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a retinal scan to check for signs of Diabetic Retinopathy</div>", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("📤 Upload Retinal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    predictions = model.predict(img_array)
    output = np.argmax(predictions)

    if output == 0:
        st.markdown(
            "<div class='result-box' style='background-color:#ffdddd; color:#d9534f;'>⚠️ Diabetic Retinopathy Detected</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box' style='background-color:#ddffdd; color:#28a745;'>✅ No Diabetic Retinopathy</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ for Healthcare AI</p>", unsafe_allow_html=True)
