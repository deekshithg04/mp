# src/ai_recommendation.py
import os
import json
import re

FALLBACK_DB = {
    "Tomato Mosaic Virus": {
        "Remedies": [
            "Remove infected leaves.",
            "Disinfect tools."
        ],
        "Preventive": [
            "Use certified seeds.",
            "Avoid touching wet plants."
        ],
        "Fertilizer": [
            "Use balanced NPK.",
            "Add organic compost."
        ]
    }
}

GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def _make_prompt(disease_name, crop_type):
    return f"""
Return STRICT JSON ONLY with:
{{
  "Remedies": [],
  "Preventive": [],
  "Fertilizer": []
}}

Disease: {disease_name}
Crop: {crop_type}

Rules:
- No extra text outside JSON.
- Max 5 items per list.
- Very simple language.
"""


def get_ai_recommendations(disease_name, crop_type="general crop"):

    # ❌ Do NOT give AI outputs for unknown
    if disease_name.lower().startswith("unknown"):
        return {
            "Remedies": ["Cannot generate remedy for unknown leaf."],
            "Preventive": [],
            "Fertilizer": []
        }

    key = disease_name.title()

    # fallback db
    if key in FALLBACK_DB:
        return FALLBACK_DB[key]

    if not GEMINI_KEY:
        return {
            "Remedies": [f"No AI available for {disease_name}."],
            "Preventive": [],
            "Fertilizer": []
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)

        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = _make_prompt(disease_name, crop_type)

        res = model.generate_content(prompt)
        text = res.text

        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())

        return {"Remedies": [text], "Preventive": [], "Fertilizer": []}

    except Exception as e:
        return {
            "Remedies": [f"AI Error: {e}"],
            "Preventive": [],
            "Fertilizer": []
        }