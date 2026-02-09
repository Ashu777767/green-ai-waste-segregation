import tensorflow as tf
from tensorflow.keras import layers, models
#“The U-Net model consists of an encoder–decoder architecture with skip connections and approximately 0.48 million trainable parameters.”
IMG_SIZE = 128
NUM_CLASSES = 3  # background, plastic, organic

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def unet_model():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    bn = conv_block(p3, 128)

    # Decoder
    u3 = layers.UpSampling2D()(bn)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 64)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 32)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 16)

    outputs = layers.Conv2D(NUM_CLASSES, 1, activation="softmax")(c6)

    model = models.Model(inputs, outputs)
    return model
