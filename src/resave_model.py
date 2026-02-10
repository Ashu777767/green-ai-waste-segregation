import tensorflow as tf

# OLD model (jo abhi crash kar raha hai)
model = tf.keras.models.load_model(
    "outputs/models/waste_classifier.keras",
    compile=False
)

# Re-save in legacy HDF5 format
model.save("outputs/models/waste_classifier.h5")
print("✅ Model re-saved as .h5")
