import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# =====================
# CONFIG
# =====================
IMG_SIZE = (224, 224)
MODEL_PATH = "outputs/models/waste_classifier.keras"

# =====================
# LOAD MODEL
# =====================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# =====================
# START CAMERA
# =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("❌ Camera not accessible")

print("🎥 Camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame (BGR → RGB → PIL)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_resized = img_pil.resize(IMG_SIZE)

    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_input, verbose=0)[0][0]

    if pred > 0.5:
        label = "Non-Biodegradable"
        color = (0, 0, 255)  # Red
        confidence = pred
    else:
        label = "Biodegradable"
        color = (0, 255, 0)  # Green
        confidence = 1 - pred

    text = f"{label} ({confidence*100:.2f}%)"

    # Draw label
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Waste Classification - Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped")
