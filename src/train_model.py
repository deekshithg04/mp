# src/train_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import itertools
import os
from sklearn.metrics import confusion_matrix

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DATASET = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 6         # change as needed
THRESHOLD = 0.60   # not used here but saved for reference
# ----------------------------

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

def plot_and_save_history(history, out_prefix):
    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_prefix + "_accuracy.png", dpi=150)
    plt.close()

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_prefix + "_loss.png", dpi=150)
    plt.close()

def plot_and_save_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# Train per-crop
for crop_dir in sorted(BASE_DATASET.iterdir()):
    if not crop_dir.is_dir():
        continue
    crop_name = crop_dir.name
    # Skip any 'crop_identifier' folder if present
    if crop_name.lower() == "crop_identifier":
        continue

    print(f"\n=== Training for crop: {crop_name} ===")

    train_gen = datagen.flow_from_directory(
        crop_dir, target_size=IMG_SIZE, batch_size=BATCH, subset="training", shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        crop_dir, target_size=IMG_SIZE, batch_size=BATCH, subset="validation", shuffle=False
    )

    # Save class names (index order)
    classes = {v: k for k, v in train_gen.class_indices.items()}
    classes_list = [classes[i] for i in range(len(classes))]
    with open(MODEL_DIR / f"{crop_name.lower()}_classes.txt", "w") as f:
        for c in classes_list:
            f.write(c + "\n")

    # Build model
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE,3))
    base.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(train_gen.num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    model_path = MODEL_DIR / f"{crop_name.lower()}_model.h5"
    model.save(model_path)
    print(f"Saved model -> {model_path}")

    # Save training history
    with open(MODEL_DIR / f"{crop_name.lower()}_history.json", "w") as hf:
        json.dump(history.history, hf)

    # Plots
    out_prefix = str(MODEL_DIR / f"{crop_name.lower()}")
    plot_and_save_history(history, out_prefix)

    # Confusion matrix on validation set
    # get predictions on val set
    val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
    y_true = val_gen.classes
    preds = model.predict(val_gen, steps=val_steps, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    plot_and_save_confusion(y_true, y_pred, classes_list, out_prefix + "_confusion.png")

    # Validation accuracy summary
    val_acc = history.history.get('val_accuracy', [])[-1] if history.history.get('val_accuracy') else None
    print(f"{crop_name} - Latest validation accuracy: {val_acc}")

print("\nAll crop models trained.")