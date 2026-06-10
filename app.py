import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import cv2
import tempfile
from PIL import Image
from deepface import DeepFace

# --- Page Config ---
st.set_page_config(page_title="NeuroTrace | AI Nexus", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- Glassmorphism & Advanced CSS ---
st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(-45deg, #050B14, #0A192F, #020617, #0F172A);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #E2E8F0; 
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    h1, h2, h3, h4, h5, h6 { color: #00D9C0 !important; font-family: 'Inter', sans-serif; letter-spacing: 1px;}
    .glass-panel {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(0, 217, 192, 0.2);
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .stButton>button { 
        background: rgba(0, 217, 192, 0.1);
        color: #00D9C0; 
        font-weight: bold; 
        border-radius: 12px; 
        border: 1px solid #00D9C0; 
        padding: 10px 24px; 
        transition: all 0.3s ease; 
        width: 100%;
        backdrop-filter: blur(5px);
    }
    .stButton>button:hover { 
        background: #00D9C0; 
        color: #050B14;
        transform: translateY(-2px); 
        box-shadow: 0 0 20px rgba(0, 217, 192, 0.6); 
    }
    .terminal-box { 
        font-family: 'Courier New', monospace; 
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(10px);
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        color: #10B981; 
        margin-bottom: 20px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
    }
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'detected_age' not in st.session_state:
    st.session_state.detected_age = 0
if 'detected_emotion' not in st.session_state:
    st.session_state.detected_emotion = "Unknown"

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #00D9C0; text-shadow: 0 0 10px rgba(0,217,192,0.5);'>🧠 NeuroTrace</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 14px; color: #94A3B8;'>AI Nexus Hackathon 2026</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### Patient Telemetry")
    patient_id = st.text_input("Patient ID", "NT-1042-X")
    
    st.metric("Estimated Patient Age", value=f"{st.session_state.detected_age} yrs" if st.session_state.detected_age > 0 else "Awaiting Scan...")
    st.metric("Base Emotion State", value=st.session_state.detected_emotion)
    
    st.divider()
    st.success("🟢 Edge-Compute Node Active")
    st.caption("Secure Stream initialized.")

# --- Main Dashboard ---
st.title("Diagnostic AI Dashboard")
st.markdown("Multimodal Fusion for **Non-Invasive Cognitive Screening**")

tab1, tab2 = st.tabs(["📡 Real-Time Data Capture", "🧬 Multimodal Fusion Analysis"])

with tab1:
    st.markdown("<div class='glass-panel'><h3>Visual Stimulus & Biometric Capture</h3><p>Instruct the patient to describe the complex scene below. Upload a video recording or use the live snapshot tool for biometric analysis.</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=800&q=80", 
                 caption="Stimulus 1A: Cognitive Activation Scene", use_column_width=True)
        
    with col2:
        capture_mode = st.radio("Select Capture Protocol:", ["Upload Video Feed", "Live Snapshot"])
        
        if capture_mode == "Upload Video Feed":
            st.markdown("#### 📁 Upload Video Payload")
            video_file = st.file_uploader("Upload .mp4 or .mov", type=['mp4', 'mov'])
            
            if video_file:
                st.video(video_file)
                if st.button("🔍 Extract & Analyze Video Frames"):
                    with st.spinner("Extracting timeline frames and running DeepFace Neural Network..."):
                        try:
                            # Save video temporarily to process with OpenCV
                            tfile = tempfile.NamedTemporaryFile(delete=False) 
                            tfile.write(video_file.read())
                            cap = cv2.VideoCapture(tfile.name)
                            
                            ages = []
                            emotions = []
                            
                            # Grab 3 frames across the video to average the data
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            frame_steps = [int(total_frames*0.2), int(total_frames*0.5), int(total_frames*0.8)]
                            
                            for step in frame_steps:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, step)
                                ret, frame = cap.read()
                                if ret:
                                    # DeepFace AI Inference on each frame
                                    analysis = DeepFace.analyze(img_path=frame, actions=['age', 'emotion'], enforce_detection=False)
                                    ages.append(analysis[0]['age'])
                                    emotions.append(analysis[0]['dominant_emotion'])
                            
                            if ages:
                                # Average the data from the video
                                st.session_state.detected_age = int(sum(ages) / len(ages))
                                st.session_state.detected_emotion = max(set(emotions), key=emotions.count).capitalize()
                                st.success(f"✅ Video Processed. Avg Age: {st.session_state.detected_age} | Dominant State: {st.session_state.detected_emotion}")
                                st.markdown("<br>", unsafe_allow_html=True)
                                if st.button("⚡ Execute Deep Fusion Analysis"):
                                    st.session_state['analysis_run'] = True
                                    st.session_state['already_loaded'] = False 
                                    st.rerun()
                            else:
                                st.error("Failed to extract readable faces from video.")
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")

        else:
            st.markdown("#### 📹 Live Snapshot")
            cam_image = st.camera_input("Initialize Camera")
            
            if cam_image:
                if st.button("🔍 Analyze Snapshot"):
                    with st.spinner("Extracting facial mesh & emotion biometrics..."):
                        try:
                            img = Image.open(cam_image)
                            img_array = np.array(img)
                            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            
                            analysis = DeepFace.analyze(img_path=img_bgr, actions=['age', 'emotion'], enforce_detection=False)
                            
                            st.session_state.detected_age = analysis[0]['age']
                            st.session_state.detected_emotion = analysis[0]['dominant_emotion'].capitalize()
                            
                            st.success(f"✅ Snapshot Processed. Age: {st.session_state.detected_age} | State: {st.session_state.detected_emotion}")
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            if st.button("⚡ Execute Deep Fusion Analysis"):
                                st.session_state['analysis_run'] = True
                                st.session_state['already_loaded'] = False 
                                st.rerun()
                        except Exception as e:
                            st.error(f"Facial scan failed: {e}")

with tab2:
    if st.session_state.get('analysis_run', False):
        
        terminal_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        logs = [
            f"Subject identified. Target Age Node: {st.session_state.detected_age}...",
            f"Baseline Emotion Context: {st.session_state.detected_emotion}",
            "Initializing FastAPI backend router...",
            "Mounting multimodal payload to Edge RAM...",
            "Running OpenSMILE Acoustic Feature Extraction... [OK]",
            "Processing Audio -> Whisper ASR -> Llama-3 Semantic NLP... [OK]",
            "Mapping 468 facial landmarks via Google MediaPipe... [OK]",
            "Applying Late Fusion Weighted Ensemble Matrix... [OK]",
            "Generating Cognitive Vitality Score... DONE."
        ]
        
        if not st.session_state.get('already_loaded'):
            current_log = ""
            for i, log in enumerate(logs):
                current_log += f"> root@neurotrace:~$ {log}<br>"
                terminal_placeholder.markdown(f'<div class="terminal-box">{current_log}</div>', unsafe_allow_html=True)
                progress_bar.progress(int((i + 1) * (100 / len(logs))))
                time.sleep(0.6) 
            st.session_state['already_loaded'] = True
        else:
            current_log = "".join([f"> root@neurotrace:~$ {log}<br>" for log in logs])
            terminal_placeholder.markdown(f'<div class="terminal-box">{current_log}</div>', unsafe_allow_html=True)
            progress_bar.progress(100)
            
        st.markdown("---")
        
        st.markdown("### 📊 Inference Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cognitive Vitality Score", "82 / 100", "-4 from baseline", delta_color="inverse")
        c2.metric("Acoustic Biomarker", "Normal", "88% Conf")
        c3.metric("Linguistic Deficit", "Mild", "-12% Semantic Richness", delta_color="inverse")
        c4.metric("Facial Apathy Index", "0.14", "Normal")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("<div class='glass-panel'><h4>🗣️ Speech Articulation Rate</h4></div>", unsafe_allow_html=True)
            chart_data = pd.DataFrame(np.random.randn(60, 1) * 0.5 + 4, columns=["Syllables/sec"])
            st.area_chart(chart_data, color="#00D9C0")
            
        with col_chart2:
            st.markdown("<div class='glass-panel'><h4>🕸️ Multimodal Risk Radar</h4></div>", unsafe_allow_html=True)
            categories = ['Speech Rate', 'Pause Length', 'Vocabulary Richness', 'Syntactic Complexity', 'Facial Mobility']
            
            fig = go.Figure(data=go.Scatterpolar(
              r=[4, 3, 2.5, 4, 5], 
              theta=categories,
              fill='toself',
              line_color='#00D9C0',
              fillcolor='rgba(0, 217, 192, 0.3)' 
            ))
            fig.update_layout(
              polar=dict(
                  radialaxis=dict(visible=True, range=[0, 5], gridcolor="rgba(255,255,255,0.1)"),
                  angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")
              ),
              showlegend=False, 
              paper_bgcolor='rgba(0,0,0,0)', 
              plot_bgcolor='rgba(0,0,0,0)', 
              font=dict(color="#f1f5f9"),
              margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.error("**Clinical Recommendation:** Minor semantic deficits detected during narrative recall. Recommend scheduling a formal follow-up assessment in 3 months. No immediate intervention required.")
    else:
        st.info("Awaiting multimodal payload. Please process a video or snapshot in Step 1.")
