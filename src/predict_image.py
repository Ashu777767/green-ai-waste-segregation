import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


# =====================
# CONFIG
# =====================
IMG_SIZE = (224, 224)
MODEL_PATH = "outputs/models/waste_classifier.keras"
IMAGE_NAME = "test.jpg"

# =====================
# PROJECT ROOT
# =====================
PROJECT_ROOT = os.getcwd()
IMG_PATH = os.path.join(PROJECT_ROOT, IMAGE_NAME)

print("📂 Image path:", IMG_PATH)

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError("❌ test.jpg not found in project root!")

# =====================
# LOAD MODEL
# =====================
model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# LOAD IMAGE (PIL – SAFE)
# =====================
img = Image.open(IMG_PATH).convert("RGB")
img_resized = img.resize(IMG_SIZE)
img_array = np.array(img_resized) / 255.0
img_input = np.expand_dims(img_array, axis=0)

# =====================
# PREDICT
# =====================
pred = model.predict(img_input)[0][0]

if pred > 0.5:
    label = "Non-Biodegradable 🚯"
    color = "red"
else:
    label = "Biodegradable ♻️"
    color = "green"

print("✅ Prediction:", label)
# ============================
# SMART POST-CLASSIFICATION LOGIC
# ============================

if "Non-Biodegradable" in label:
    material_guess = "Plastic / Metal / E-Waste"
    recyclable = "Possible (Material dependent)"
    bin_type = "Dry Waste / Recycling Bin"
    note = "Requires further sorting at recycling facility"
else:
    material_guess = "Food / Paper / Leaf / Wood"
    recyclable = "Compostable"
    bin_type = "Wet / Organic Waste Bin"
    note = "Can be naturally decomposed"

print("\n===== SMART GREEN AI OUTPUT =====")
print("Main Category:", label)
print("Possible Material:", material_guess)
print("Recyclable:", recyclable)
print("Suggested Bin:", bin_type)
print("Note:", note)


# =====================
# VISUALIZE
# =====================
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(
    f"{label}\n{material_guess}\nBin: {bin_type}",
    color=color
)

plt.axis("off")
plt.show()
