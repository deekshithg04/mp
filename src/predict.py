import tensorflow as tf
import numpy as np
from preprocess import preprocess_image
from pathlib import Path

# ------------------------------
# Determine the absolute path to the model
BASE_DIR = Path(__file__).resolve().parent.parent  # points to 'mp' folder
MODEL_PATH = BASE_DIR / "models" / "crop_disease_model.h5"

# Load trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(f"Unable to load model at {MODEL_PATH}. Error: {e}")

# Class names (must match order used in training)
class_names = [
    "Pepper__bell__Bacterial_spot",
    "Pepper__bell__healthy",
    "Potato__Early_blight",
    "Potato__healthy",
    "Potato__Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# Confidence threshold to reject unknown images
CONFIDENCE_THRESHOLD = 0.60  # You can increase to 0.7 if still too lenient

def predict_disease(image):
    """
    Predict disease from PIL image
    Returns: (disease_name, confidence)
    """
    try:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        # Check confidence level
        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown or Non-trained Leaf", confidence

        disease_name = class_names[predicted_index]

        # Optional: make output simpler for farmers
        disease_simple = disease_name.replace("__", " → ").replace("_", " ")

        return disease_simple, confidence

    except Exception as e:
        return f"Prediction error: {e}", 0.0