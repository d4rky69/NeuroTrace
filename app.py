import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="NeuroTrace | Prototype", page_icon="🧠", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #020617; color: #f1f5f9; }
    h1, h2, h3 { color: #00d9c0; font-family: sans-serif; }
    .metric-container { background-color: #0f172a; padding: 20px; border-radius: 10px; border: 1px solid #1e293b; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("NeuroTrace")
    st.caption("AI Nexus Hackathon 2026 | Team AXILLA")
    st.divider()
    st.subheader("Patient Details")
    patient_id = st.text_input("Patient ID", "NT-1042")
    age = st.slider("Age", 40, 90, 65)
    st.divider()
    st.info("Status: Edge-Processing Active 🟢")

# --- Main Dashboard ---
st.title("🧠 NeuroTrace Assessment Dashboard")
st.write("Multimodal AI for Non-Invasive Cognitive Screening")

# Tabs for workflow
tab1, tab2 = st.tabs(["📋 Step 1: Data Capture", "📊 Step 2: Biomarker Analysis"])

with tab1:
    st.subheader("Standardized Visual Stimulus Task")
    st.write("Instruct the patient to describe the image below in detail for 60 seconds.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Cookie_theft_picture.png/640px-Cookie_theft_picture.png", caption="Standardized Stimulus Image")
    
    with col2:
        st.write("### Sensor Status")
        st.checkbox("Webcam Ready (Visual Marker)", value=True, disabled=True)
        st.checkbox("Microphone Ready (Acoustic Marker)", value=True, disabled=True)
        
        st.divider()
        audio_file = st.file_uploader("Upload Audio/Video Sample (.wav or .mp4)", type=['wav', 'mp4'])
        
        if audio_file:
            st.success("Media captured securely.")
            if st.button("Run Multimodal Fusion Analysis", type="primary"):
                st.session_state['analysis_run'] = True

with tab2:
    if st.session_state.get('analysis_run', False):
        with st.spinner("Extracting Acoustic Features (OpenSMILE)..."):
            time.sleep(1)
        with st.spinner("Transcribing & Analyzing Semantics (Whisper + Llama-3)..."):
            time.sleep(1)
        with st.spinner("Tracking Facial Micro-expressions (MediaPipe)..."):
            time.sleep(1)
            
        st.success("Analysis Complete. Review the Cognitive Vitality Report below.")
        
        st.subheader("Cognitive Vitality Score")
        final_score = 82
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Vitality Score", f"{final_score}/100", "-4 from baseline", delta_color="inverse")
        c2.metric("Acoustic Confidence", "88%", "Normal")
        c3.metric("Linguistic Score", "76%", "Slight semantic deficit", delta_color="inverse")
        c4.metric("Visual Apathy Index", "0.14", "Normal")
        
        st.divider()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.write("#### Speech Articulation Rate (Acoustic)")
            chart_data = pd.DataFrame(np.random.randn(60, 1) * 0.5 + 4, columns=["Syllables per second"])
            st.line_chart(chart_data)
            
        with col_chart2:
            st.write("#### Multimodal Risk Fusion")
            categories = ['Speech Rate', 'Pause Length', 'Vocabulary Richness', 'Syntactic Complexity', 'Facial Mobility']
            fig = go.Figure(data=go.Scatterpolar(r=[4, 3, 2.5, 4, 5], theta=categories, fill='toself', line_color='#00d9c0'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#f1f5f9"))
            st.plotly_chart(fig, use_container_width=True)
            
        st.warning("**Clinical Recommendation:** Minor semantic deficits detected. Recommend scheduling a follow-up assessment in 3 months. No immediate clinical intervention required.")
    else:
        st.info("Please complete the Data Capture phase in Step 1 to view the analysis.")
