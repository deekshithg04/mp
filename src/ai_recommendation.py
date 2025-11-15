# src/ai_recommendation.py
import google.generativeai as genai
import os

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_ai_recommendations(disease_name, crop_type="general crop"):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = (
            f"You are an agricultural expert helping local farmers.\n"
            f"The crop is '{crop_type}', and it is affected by '{disease_name}'.\n"
            f"Give a short and easy-to-understand explanation in simple words — "
            f"only the main remedies and fertilizer recommendations that farmers can follow easily.\n"
            f"Use simple language, short sentences, and bullet points.\n"
            f"Limit the response to 10–12 lines maximum.\n"
        )

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error fetching AI recommendations: {str(e)}"