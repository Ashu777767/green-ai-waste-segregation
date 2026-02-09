import tensorflow as tf
from data_loader import load_data
from model import unet_model

# Load data
X, y = load_data("data/images", "data/masks")

print("Loaded images:", X.shape)
print("Loaded masks:", y.shape)

# Build model
model = unet_model()
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Train model
model.fit(
    X, y,
    epochs=10,
    batch_size=2,
    validation_split=0.2
)

# Save model (folder already exists)
model.save("outputs/models/waste_unet.keras")

#“The model was successfully trained and saved for inference and deployment.”
print("Training complete & model saved!")
