import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from pathlib import Path

# Dataset path (each crop has its own subfolder)
BASE_DATASET = Path("../dataset")

# Model output folder
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)

# Common augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Train separate model for each crop
for crop_dir in BASE_DATASET.iterdir():
    if crop_dir.is_dir():
        crop_name = crop_dir.name
        print(f"\n🌿 Training model for {crop_name}...")

        # Create train/val generators
        train_gen = datagen.flow_from_directory(
            crop_dir,
            target_size=(224,224),
            batch_size=32,
            subset="training"
        )
        val_gen = datagen.flow_from_directory(
            crop_dir,
            target_size=(224,224),
            batch_size=32,
            subset="validation"
        )

        # Base model
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
        base_model.trainable = False

        # Add custom layers
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(train_gen.num_classes, activation="softmax")
        ])

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Train model
        model.fit(train_gen, validation_data=val_gen, epochs=6)

        # Save the model
        model_path = MODEL_DIR / f"{crop_name.lower()}_model.h5"
        model.save(model_path)
        print(f"✅ Saved {crop_name} model to {model_path}")