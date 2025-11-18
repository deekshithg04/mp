# src/predict.py

import tensorflow as tf
import numpy as np
from pathlib import Path
from preprocess import preprocess_image
import cv2

# =========================================================
# DIRECTORIES
# =========================================================
MODEL_DIR = Path("/Users/deekshith04/mp/models")
LEAF_MODEL_PATH = MODEL_DIR / "leaf_nonleaf_model.h5"

print("🔍 MODEL_DIR:", MODEL_DIR)

# -------------------------
# CONFIG
# -------------------------
CONFIDENCE_THRESHOLD = 0.65
HEALTHY_THRESHOLD = 0.80
LEAF_THRESHOLD = 0.40   # TEMP lowered for testing (change later to 0.60)


# =========================================================
# LOAD LEAF/NON-LEAF CLASSIFIER
# =========================================================
leaf_detector = None
try:
    if LEAF_MODEL_PATH.exists():
        leaf_detector = tf.keras.models.load_model(LEAF_MODEL_PATH)
        print("🌿 Leaf/Non-Leaf model loaded:", LEAF_MODEL_PATH.name)
    else:
        print("⚠️ Leaf model NOT found:", LEAF_MODEL_PATH)
except Exception as e:
    print("⚠️ ERROR loading leaf detector:", e)
    leaf_detector = None


def is_leaf_image(pil_img):
    """Return True if the image is a leaf using the CNN classifier."""

    if leaf_detector is None:
        print("⚠️ Leaf detector missing — allowing all images.")
        return True

    arr = preprocess_image(pil_img)   # (1,224,224,3)
    pred = float(leaf_detector.predict(arr, verbose=0)[0][0])

    # pred = P(nonleaf)
    leaf_prob = 1 - pred              # convert to P(leaf)

    print(f"Leaf probability: {leaf_prob:.4f}")

    # Decide if leaf
    return leaf_prob >= LEAF_THRESHOLD

# =========================================================
# LOAD PER-CROP MODELS
# =========================================================
_models = {}
_classes = {}

model_files = list(MODEL_DIR.glob("*_model.h5"))

for model_file in model_files:

    # Skip leaf classifier
    if model_file.name == "leaf_nonleaf_model.h5":
        continue

    crop = model_file.stem.replace("_model", "").lower()
    print(f"📌 Loading model: {crop}")

    try:
        _models[crop] = tf.keras.models.load_model(model_file)
    except Exception as e:
        print(f"⚠️ Error loading {model_file}: {e}")
        continue

    # Load class labels
    classes_file = MODEL_DIR / f"{crop}_classes.txt"
    if classes_file.exists():
        with open(classes_file, "r") as f:
            _classes[crop] = [x.strip() for x in f.readlines() if x.strip()]
    else:
        n = _models[crop].output_shape[-1]
        _classes[crop] = [f"class_{i}" for i in range(n)]

print("✅ Loaded Crop Models:", list(_models.keys()))


# =========================================================
# FORMAT CLEAN LABELS
# =========================================================
def format_disease_name(name):
    name = name.replace("__", " ").replace("_", " ")
    for bad in ["→", ":", "-", ">"]:
        name = name.replace(bad, " ")
    return " ".join([x for x in name.split() if x]).title()


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================
def predict_disease(image, forced_crop=None):

    # FIRST — Leaf detector check
    if not is_leaf_image(image):
        return "Not a Leaf Image", 0.0, None, None

    img = preprocess_image(image)

    # -----------------------------------------------------
    # FORCED CROP SELECTED BY USER
    # -----------------------------------------------------
    if forced_crop:
        key = forced_crop.strip().lower()
        lookup = {c.lower(): c for c in _models.keys()}

        if key not in lookup:
            return f"Invalid Crop: {forced_crop}", 0.0, None, None

        crop = lookup[key]
        model = _models[crop]
        labels = _classes[crop]

        preds = model.predict(img, verbose=0)
        conf = float(np.max(preds))
        idx = int(np.argmax(preds))
        raw = labels[idx]

        if conf < CONFIDENCE_THRESHOLD:
            return "Unknown Leaf / Not Clear", conf, crop, raw

        if "healthy" in raw.lower():
            return "Healthy Leaf", conf, crop, raw

        return format_disease_name(raw), conf, crop, raw

    # -----------------------------------------------------
    # AUTO-DETECT CROP — Choose highest confidence model
    # -----------------------------------------------------
    best_conf = 0
    best_crop = None
    best_label = None

    for crop, model in _models.items():
        preds = model.predict(img, verbose=0)
        conf = float(np.max(preds))
        idx = int(np.argmax(preds))
        label = _classes[crop][idx]

        if conf > best_conf:
            best_conf = conf
            best_crop = crop
            best_label = label

    if best_conf < CONFIDENCE_THRESHOLD:
        return "Unknown Leaf / Not Clear", best_conf, None, None

    if "healthy" in best_label.lower():
        return "Healthy Leaf", best_conf, best_crop, best_label

    return format_disease_name(best_label), best_conf, best_crop, best_label