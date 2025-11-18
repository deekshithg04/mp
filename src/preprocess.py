# src/preprocess.py
from PIL import Image
import numpy as np

def preprocess_image(image, target_size=(224,224)):
    """
    Resize, convert to RGB, normalize and return shape (1,H,W,3)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr