import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image

from utils.inference import predict, load_model, preprocess
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap
from utils.analysis import analyze_prediction
from agent.explainer import generate_report

st.set_page_config(page_title="PneumoSight AI", layout="centered")

st.title("🧠 PneumoSight AI")
st.caption("AI-powered Pneumonia Screening & Decision Support")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])


# 🔥 CACHE GPT (VERY IMPORTANT)
@st.cache_data(show_spinner=False)
def cached_report(label, confidence, risk, status):
    return generate_report(label, confidence, risk, status)


if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # 🔹 Prediction with loading
    with st.spinner("Analyzing X-ray..."):
        label, prob = predict(img)
        confidence, risk, status = analyze_prediction(label, prob)

    st.subheader("🧾 Clinical Output")

    # 🔹 Clinical display
    st.write(f"**Diagnosis:** {label}")
    st.write(f"**Confidence:** {confidence}%")
    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Status:** {status}")

    # 🔥 Confidence bar
    st.progress(int(confidence))

    # 🔥 Color-coded alert
    if label == "Pneumonia":
        st.error(f"⚠️ Pneumonia detected ({confidence}%)")
    else:
        st.success(f"✅ Normal ({confidence}%)")

    # 🔥 Uncertainty handling
    if status == "Uncertain":
        st.warning("⚠️ Model is uncertain. Further clinical testing recommended.")

    # 🔹 Grad-CAM
    st.subheader("🔍 Model Explanation (Grad-CAM)")

    model = load_model()
    img_array = preprocess(img)

    heatmap = make_gradcam_heatmap(
        model,
        img_array,
        "block14_sepconv2_act"
    )

    result_img = overlay_heatmap(img, heatmap)

    st.image(
        result_img,
        caption="Highlighted regions influencing prediction",
        use_container_width=True
    )

    # 🔹 AI Report
    st.subheader("🧠 AI Radiology Report")

    report = cached_report(label, confidence, risk, status)
    st.write(report)

    # 🔹 Safety disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ This is an AI-based screening tool and not a medical diagnosis. "
        "Please consult a qualified healthcare professional."
    )