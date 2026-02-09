import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =====================
# CONFIG
# =====================
IMG_SIZE = (224, 224)
MODEL_PATH = "outputs/models/waste_classifier.keras"

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
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =====================
# PREDICTION FUNCTION
# =====================
def predict_image_pil(img_pil):
    img_resized = img_pil.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_input, verbose=0)[0][0]

    if pred > 0.5:
        return "Non-Biodegradable", pred
    else:
        return "Biodegradable", 1 - pred

# =====================
# PAGE SETUP
# =====================
st.set_page_config(
    page_title="Green AI Smart Waste Segregation",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Green AI–Based Smart Waste Segregation System")
st.markdown(
    "### Intelligent waste classification with sustainable recycling guidance 🌱"
)
st.markdown("---")

# =====================
# SIDEBAR
# =====================
mode = st.sidebar.radio(
    "Choose Mode",
    ["📤 Upload Image & Predict", "🎥 Live Camera Classification"]
)

# =========================================================
# MODE 1: IMAGE UPLOAD
# =========================================================
if mode == "📤 Upload Image & Predict":
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
    "Live camera works only on local machine. "
    "Streamlit Cloud supports image upload mode only."
)
# =========================================================
# MODE 2: LIVE CAMERA (LOCAL USE)
# =========================================================
elif mode == "🎥 Live Camera Classification":
    st.header("🎥 Live Camera Waste Classification")

    st.info(
        "• Click **Start Camera**\n"
        "• Show waste item to webcam\n"
        "• Press **Q** to stop camera"
    )

    if st.button("▶️ Start Camera"):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Camera not accessible")
        else:
            st.success("Camera started. Press Q to stop.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                label, confidence = predict_image_pil(img_pil)
                info = smart_logic(label)

                color = (0, 255, 0) if label == "Biodegradable" else (0, 0, 255)

                cv2.putText(frame, label, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

                cv2.putText(frame, f"{confidence*100:.1f}%", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.putText(frame, info["action"], (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("Green AI Waste Classification", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()
            st.info("Camera stopped")
