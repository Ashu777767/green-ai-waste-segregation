import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
MODEL_PATH = "outputs/models/waste_classifier.keras"
IMG_SIZE = (224, 224)

# =========================
# PATH HANDLING
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
IMG_PATH = os.path.join(PROJECT_ROOT, "test.jpg")

print("Loading image from:", IMG_PATH)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# LOAD IMAGE
# =========================
img_bgr = cv2.imread(IMG_PATH)

if img_bgr is None:
    raise ValueError("❌ Image not found or cannot be read!")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, IMG_SIZE)
img_norm = img_resized / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# =========================
# PREDICT
# =========================
prediction = model.predict(img_input)[0][0]

if prediction > 0.5:
    label = "Non-Biodegradable 🚯"
    color = "red"
else:
    label = "Biodegradable ♻️"
    color = "green"

print("✅ Prediction:", label)

# =========================
# VISUALIZATION (PLT)
# =========================
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title(f"Prediction: {label}", color=color)
plt.axis("off")
plt.show()
