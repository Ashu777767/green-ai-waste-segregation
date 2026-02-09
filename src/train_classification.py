import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5   # demo ke liye enough
DATA_DIR = "data"
MODEL_PATH = "outputs/models/waste_classifier.keras"

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =========================
# MODEL (TRANSFER LEARNING)
# =========================
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Green AI (less compute)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# =========================
# SAVE MODEL
# =========================
os.makedirs("outputs/models", exist_ok=True)
model.save(MODEL_PATH)

print("\n✅ Training complete!")
print(f"✅ Model saved at: {MODEL_PATH}")
print("♻️ Waste Classification Model Ready")
