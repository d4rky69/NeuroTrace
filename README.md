# 🧠 NeuroTrace 

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![Status](https://img.shields.io/badge/Status-Hackathon_Prototype-success?style=for-the-badge)

**Multimodal AI for Non-Invasive Cognitive Screening** *Submission for the AI Nexus Hackathon 2026 by Team AXILLA*

## 📖 Overview
Over 75% of global dementia cases remain undiagnosed until irreversible damage occurs. **NeuroTrace** transforms standard consumer hardware (webcams and microphones) into a clinical-grade cognitive assessment tool. By analyzing three concurrent modalities—**Acoustic, Linguistic, and Visual**—during a 60-second automated storytelling task, NeuroTrace detects early biomarkers of cognitive decline at zero marginal cost.

## 🚀 Features (Prototype)
* **Standardized Stimulus Task:** Digital administration of the "Cookie Theft" visual description test.
* **Simulated Multimodal Fusion:** Demonstration of concurrent feature extraction across speech rate, semantic richness, and facial micro-expressions.
* **Cognitive Vitality Dashboard:** Real-time visual generation of risk scores and clinical recommendations via radar and line charting.

## 🛠️ Tech Stack Architecture 
*(Note: Current repository contains the Level 1 Frontend UI Prototype. Backend inference logic is abstracted for this phase).*
* **Frontend:** Streamlit, React.js (Planned)
* **Backend:** FastAPI, Python
* **AI/ML Modalities:** * *Acoustic:* OpenSMILE
  * *Linguistic:* OpenAI Whisper ASR + Llama-3 (Groq API)
  * *Visual:* MediaPipe Face Mesh, OpenCV

## 💻 How to Run the Prototype Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/d4rky69/NeuroTrace.git](https://github.com/d4rky69/NeuroTrace.git)
   cd NeuroTrace
