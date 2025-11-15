# preprocess.py
from PIL import Image
import numpy as np

def preprocess_image(image, target_size=(224,224)):
    """
    Resize and normalize image for CNN model
    """
    img = image.resize(target_size)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array