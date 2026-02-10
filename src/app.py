import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================
# CONFIG
# =====================
IMG_SIZE = (224, 224)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "outputs", "models", "waste_classifier.keras"
)



# =====================
# SMART GREEN LOGIC
# =====================
def smart_logic(label):
    if label == "Non-Biodegradable":
        return {
            "possible_items": "Plastic bottles, Plastic bags, Metal cans, E-waste",
            "recyclable": "Yes (Plastic & Metal), Special process for E-waste",
            "action": "Dry Waste / Recycling Bin",
            "note": "Further sorting required at recycling facility"
        }
    else:
        return {
            "possible_items": "Food waste, Paper, Leaves, Wood",
            "recyclable": "Yes (Compost / Paper recycling)",
            "action": "Wet / Organic Waste Bin",
            "note": "Eco-friendly disposal through composting"
        }

# =====================
# LOAD MODEL (CLOUD SAFE)
# =====================
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
    except Exception as e:
        st.error("❌ Failed to load ML model")
        st.exception(e)
        st.stop()

model = load_model()

# =====================
# PREDICTION FUNCTION
# =====================
def predict_image_pil(img_pil):
    img_resized = img_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_input, verbose=0)
    prob = float(pred[0][0])  # binary classification output

    if prob > 0.5:
        return "Non-Biodegradable", prob
    else:
        return "Biodegradable", 1 - prob

# =====================
# PAGE SETUP
# =====================
st.set_page_config(
    page_title="Green AI Smart Waste Segregation",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Green AI–Based Smart Waste Segregation System")
st.markdown("### Intelligent waste classification with sustainable recycling guidance 🌱")
st.markdown("---")

# =====================
# SIDEBAR
# =====================
mode = st.sidebar.radio(
    "Choose Mode",
    ["📤 Upload Image & Predict"]
)

# =========================================================
# IMAGE UPLOAD MODE (CLOUD + LOCAL)
# =========================================================
st.header("📤 Upload Waste Image")

uploaded_file = st.file_uploader(
    "Choose a waste image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Waste"):
        label, confidence = predict_image_pil(img)
        info = smart_logic(label)

        st.markdown("---")
        st.markdown(f"## 🗑️ {label}")
        st.progress(int(confidence * 100))
        st.caption(f"Confidence: {confidence*100:.2f}%")

        st.markdown("### ♻️ Smart Recycling Insights")
        st.write(f"**Possible Waste Type:** {info['possible_items']}")
        st.write(f"**Recyclable:** {info['recyclable']}")
        st.write(f"**Recommended Bin:** {info['action']}")
        st.write(f"**Note:** {info['note']}")

st.warning(
    "Live camera functionality is disabled on Streamlit Cloud. "
    "This demo uses image-based classification for reliable deployment."
)
