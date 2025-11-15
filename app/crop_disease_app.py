# crop_disease_app.py
import streamlit as st
from PIL import Image
import sys
from pathlib import Path

# Add src folder to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from predict import predict_disease
from ai_recommendation import get_ai_recommendations

# ---------------------------------
# Helper to clean disease name
# ---------------------------------
def format_disease_name(name):
    # Replace underscores and double underscores with spaces
    name = name.replace("__", " ").replace("_", " ").strip()
    
    # Remove unwanted symbols (like →, :, etc.)
    for symbol in ["→", ":", "-", ">"]:
        name = name.replace(symbol, " ")

    # Split into words
    parts = name.split()
    
    # Remove duplicate consecutive words (like "Tomato Tomato")
    cleaned_parts = []
    for word in parts:
        if not cleaned_parts or cleaned_parts[-1].lower() != word.lower():
            cleaned_parts.append(word)
    
    # Join back and title-case
    formatted = " ".join(cleaned_parts).title()
    return formatted

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="🌿 Crop Disease Detection", layout="wide")

st.title("🌿 AI Crop Disease Detection System")
st.write("Upload leaf images to get disease predictions and AI-generated recommendations using Google Gemini Pro.")

crop_type = st.text_input("Enter Crop Type (optional)", "general crop")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=False)

    with st.spinner("🔍 Predicting disease..."):
        predicted_disease, confidence = predict_disease(image)

    formatted_disease = format_disease_name(predicted_disease)
    st.success(f"**Disease Detected:** {formatted_disease}")
    st.info(f"**Accuracy:** {confidence*100:.2f}%")

    with st.spinner("🤖 Fetching AI-based recommendations..."):
        recommendations = get_ai_recommendations(formatted_disease, crop_type)

    st.markdown("### 🌱 Remedies & Fertilizer Recommendations:")
    st.write(recommendations)