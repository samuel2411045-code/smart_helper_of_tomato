import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3
NUM_CLASSES = 8

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "Variant-a(Multiclass Classification)")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

def build_cnn():
    """Build a CNN model using MobileNetV2 for transfer learning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_vit():
    """Build a simplified Vision Transformer (ViT) model."""
    input_shape = (*IMG_SIZE, 3)
    patch_size = 32
    num_patches = (IMG_SIZE[0] // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_layers = 4
    mlp_head_units = [128, 64]

    inputs = layers.Input(shape=input_shape)
    
    # 1. Patch Creation & Embedding
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)
    
    # 2. Positional Embedding
    # Create trainable positional embeddings
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(tf.range(num_patches))
    # Standard broadcasting: (batch, 49, 64) + (49, 64) = (batch, 49, 64)
    encoded_patches = layers.Add()([patches, pos_embed])

    # 3. Transformer Layers
    for _ in range(transformer_layers):
        # Layer Normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head Attention (using two inputs for query and value, Keras handles key=value)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip Connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer Normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(projection_dim, activation="gelu")(x3)
        x3 = layers.Dropout(0.1)(x3)
        # Skip Connection 2
        encoded_patches = layers.Add()([x3, x2])

    # 4. Classification Head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    for units in mlp_head_units:
        representation = layers.Dense(units, activation="gelu")(representation)
        representation = layers.Dropout(0.1)(representation)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(representation)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model_type):
    print(f"\n--- Starting Training for {model_type} ---")
    
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if model_type == "CNN":
        model = build_cnn()
        save_path = os.path.join(models_dir, "disease_model.h5")
    else:
        model = build_vit()
        save_path = os.path.join(models_dir, "disease_vit_model.h5")

    os.makedirs(models_dir, exist_ok=True)


    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["CNN", "ViT"], required=True)
    args = parser.parse_args()
    
    train_model(args.model)
