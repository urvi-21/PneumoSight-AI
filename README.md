# 🧠 PneumoSight AI

**AI-Powered Pneumonia Screening & Clinical Decision Support System**

PneumoSight AI is an end-to-end deep learning system that detects pneumonia from chest X-rays and provides clinically interpretable insights using Grad-CAM and AI-generated radiology reports.

---

## 🧩 Problem

Pneumonia is a serious respiratory condition requiring early and accurate diagnosis.  
Manual interpretation of chest X-rays is time-consuming and subject to variability, especially in resource-constrained settings.

---

## 💡 Solution

PneumoSight AI automates pneumonia detection and enhances decision-making by combining:

- Deep learning-based image classification  
- Explainable AI (Grad-CAM) for visual interpretability  
- Confidence-based risk stratification  
- AI-generated radiology reports  

---

## ⚙️ Key Features

### 🩺 Pneumonia Detection
- Binary classification (Normal / Pneumonia)
- Transfer learning using Xception

### 📊 Clinical Decision Support
- Confidence-aware predictions  
- Risk stratification (Low / Moderate / High)  
- Uncertainty handling  

### 🔍 Explainability (Grad-CAM)
- Highlights lung regions influencing predictions  
- Improves transparency and trust  

### 🧠 AI Radiology Report
- Automatically generates:
  - Findings  
  - Impression  
  - Recommendations  

### ⚠️ Safety Layer
- Includes medical disclaimer  
- Encourages professional consultation  

---

## 📊 Model Performance

- **Accuracy:** 95.7%  
- **Precision:** 99.08%  
- **Recall:** 95.28%  
- **F1 Score:** 97.14%  

> High recall ensures minimal false negatives, which is critical in medical diagnosis systems.

---

## 🏗️ System Architecture
```
User Upload (X-ray)
↓
Deep Learning Model (Xception)
↓
Prediction + Confidence
↓
Risk Analysis Engine
↓
Grad-CAM Visualization
↓
AI Radiology Report (LLM)
```


---

## 🧪 Model Details

- Architecture: Xception (Transfer Learning)  
- Input Size: 224 × 224  
- Output: Binary classification  
- Fine-tuned final layers  

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow, Keras  
- **Frontend:** Streamlit  
- **Explainability:** Grad-CAM  
- **AI Integration:** OpenAI API  
- **Image Processing:** OpenCV, PIL  

---

## Demo

│<img width="573" height="753" alt="Screenshot 2026-04-08 180943" src="https://github.com/user-attachments/assets/708dc196-bd89-449e-ad4d-accbea65d2b5" />

<img width="523" height="724" alt="Screenshot 2026-04-08 180929" src="https://github.com/user-attachments/assets/133cb8a8-c75a-4457-91b9-0c7193685e4a" />


---


## 📁 Project Structure

```
PneumoSight-AI/
│
├── app/
│   └── app.py                
│
├── utils/
│   ├── inference.py         
│   ├── gradcam.py          
│   ├── analysis.py         
│
├── agent/
│   └── explainer.py         
│
├── models/                  
│
├── requirements.txt         
├── .gitignore               
└── README.md                
```
