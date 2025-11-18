# app/crop_disease_app.py

import base64
from io import BytesIO
import streamlit as st
from PIL import Image
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from predict import predict_disease
from ai_recommendation import get_ai_recommendations


# -------------------------------------------------------
# SMALL IMAGE DISPLAY FUNCTION
# -------------------------------------------------------
def show_small_image(img, size=200, caption="Selected Image"):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:10px;">
            <img src="data:image/png;base64,{encoded}"
                 style="width:{size}px; height:{size}px;
                        object-fit:contain;
                        border-radius:10px;
                        border:1px solid #ccc;
                        padding:5px;">
            <p style="color:gray; font-size:14px;">{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------------
st.set_page_config(page_title="Crop Disease Detection", layout="wide")
st.title("🌿 Crop Disease Detection — Per-Crop Model System")


# -------------------------------------------------------
# CAMERA SIZE FIX (CSS OVERRIDE)
# -------------------------------------------------------
st.markdown("""
<style>
    div[data-testid="stCameraInput"] {
        width: 100% !important;
        max-width: 500px !important;
        min-height: 420px !important;
        padding: 20px !important;
        margin: auto !important;

        border: 2px solid #ddd !important;
        border-radius: 15px !important;
        background: #fafafa !important;

        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }

    div[data-testid="stCameraInput"] video {
        width: 250px !important;
        height: 250px !important;
        object-fit: cover !important;
        border-radius: 12px !important;
    }

    div[data-testid="stCameraInput"] canvas {
        width: 250px !important;
        height: 250px !important;
        object-fit: cover !important;
        border-radius: 12px !important;
    }

    div[data-testid="stCameraInput"] button {
        margin-top: 15px !important;
        width: 90% !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# INPUT SECTION
# -------------------------------------------------------
st.header("Upload or Capture Leaf Image")

mode = st.radio("Choose Input Mode", ["Upload Image", "Camera Capture"])
crop_hint = st.text_input("Crop type (optional, helps AI recommendations)")

img = None

# Upload mode
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        show_small_image(img, size=180, caption="Uploaded Image")

# Camera mode
else:
    st.write("📷 Take a picture below:")
    cam_img = st.camera_input("")
    if cam_img:
        img = Image.open(cam_img)
        show_small_image(img, size=180, caption="Captured Image")


# -------------------------------------------------------
# PREDICTION + AI RECOMMENDATIONS
# -------------------------------------------------------
if img and st.button("Predict Disease"):
    with st.spinner("Analyzing leaf..."):
        disease, conf, crop, raw = predict_disease(img)

    # --------------------------------------
    # CASE 1: NOT A LEAF IMAGE
    # --------------------------------------
    if disease == "Not a Leaf Image":
        st.error("🚫 This is NOT a leaf image.")
        st.info(f"🔍 Confidence: {conf * 100:.2f}% sure it is NOT a leaf.")
        st.warning("Please upload a clear leaf image.")
        st.stop()

    # --------------------------------------
    # CASE 2: NORMAL LEAF OUTPUT
    # --------------------------------------
    st.success(f"🌱 **Disease Detected:** {disease}")
    st.info(f"📊 **Confidence:** {conf * 100:.2f}%")

    if crop:
        st.write(f"📌 Model Used: **{crop.title()}**")

    # ----------- AI Recommendations -----------
    st.subheader("🧠 AI Recommendations")

    rec = get_ai_recommendations(
        disease,
        crop_hint if crop_hint else (crop or "general crop")
    )

    st.markdown("### 🌿 Remedies")
    for r in rec.get("Remedies", []):
        st.write(f"- {r}")

    st.markdown("### 🛡️ Preventive Measures")
    for p in rec.get("Preventive", []):
        st.write(f"- {p}")

    st.markdown("### 🧪 Fertilizer Suggestions")
    for f in rec.get("Fertilizer", []):
        st.write(f"- {f}")