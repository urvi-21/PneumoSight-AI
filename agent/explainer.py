from openai import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st

# 🔹 Load environment once
load_dotenv()

# 🔹 Create client once
api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

def generate_report(label, confidence, risk, status):
    try:
        prompt = f"""
You are an AI radiology assistant.

Patient Chest X-ray Analysis:

Diagnosis: {label}
Confidence: {confidence}%
Risk Level: {risk}
Status: {status}

Generate a structured clinical report in EXACT format:

Findings:
- Bullet points describing observed patterns

Impression:
- Clear medical conclusion with likelihood

Recommendation:
- Actionable next steps

Keep it concise, realistic, and clinically relevant.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        # 🔥 SMART FALLBACK (context-aware)
        if label == "Pneumonia":
            return f"""
Findings:
- Possible lung opacity detected

Impression:
- Likely pneumonia ({confidence}% confidence)

Recommendation:
- Immediate clinical evaluation advised
"""
        else:
            return f"""
Findings:
- No significant abnormalities detected

Impression:
- Likely normal chest X-ray ({confidence}% confidence)

Recommendation:
- Routine follow-up if needed
"""