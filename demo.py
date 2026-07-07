import streamlit as st
import asyncio
import logging
import json
import os
import time
from pathlib import Path
from workers.manager import process_videos
from app.config import config
from app.logger import logger as app_logger

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="CineScribe AI — Cinematic Video Captioning",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern premium dark-mode styling and micro-animations
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F, #FFC7C7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        animation: fadeIn 1s ease-in-out;
    }
    .subtitle {
        color: #A0A5B5;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        border-bottom: 2px solid #33363F;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Premium style caption card */
    .caption-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        border: 1px solid #2D3139;
        background: linear-gradient(145deg, #1B1E24, #121418);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    .caption-card:hover {
        transform: translateY(-3px);
        border-color: #FF4B4B;
        box-shadow: 0 8px 24px rgba(255, 75, 75, 0.15);
    }
    .style-header {
        font-size: 0.9rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    /* Style-specific header coloring */
    .style-formal { color: #5C7CFA; }
    .style-sarcastic { color: #FF922B; }
    .style-humorous_tech { color: #20C997; }
    .style-humorous_non_tech { color: #DA77F2; }
    .style-custom { color: #FCC419; }
    
    .caption-content {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #E2E8F0;
    }
    
    /* Performance metric badge */
    .metric-badge {
        background-color: #1A1D24;
        border: 1px solid #2D3139;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #20C997;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #A0A5B5;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Timeline styling */
    .timeline-item {
        border-left: 3px solid #FF4B4B;
        margin-left: 10px;
        padding-left: 20px;
        padding-bottom: 1.5rem;
        position: relative;
    }
    .timeline-item::before {
        content: '';
        width: 12px;
        height: 12px;
        background-color: #FF4B4B;
        border-radius: 50%;
        position: absolute;
        left: -8px;
        top: 4px;
        border: 2px solid #0E1117;
    }
    .timeline-time {
        font-weight: 700;
        color: #FF8F8F;
        font-size: 0.95rem;
    }
    .timeline-desc {
        color: #E2E8F0;
        margin-top: 0.25rem;
        font-size: 1rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# 2. Setup Session State
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "result" not in st.session_state:
    st.session_state["result"] = None
if "processing" not in st.session_state:
    st.session_state["processing"] = False

class StreamlitLogHandler(logging.Handler):
    """Intercepts framework logs and updates st.session_state["logs"] for real-time visualization."""
    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        log_entry = self.format(record)
        st.session_state["logs"].append(log_entry)

# 3. Sidebar Configuration Panel
st.sidebar.markdown("### 🎬 Configuration")

# Setup preset options
presets = {
    "Autumn City Boulevard": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "Cat in a Garden / Jungle": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "Custom URL": ""
}

selected_preset = st.sidebar.selectbox("Choose a Video Source", list(presets.keys()))
video_url = presets[selected_preset]

# Custom video input
if selected_preset == "Custom URL":
    video_url = st.sidebar.text_input("Paste Remote Video URL", placeholder="https://example.com/video.mp4")

# Local file upload option
uploaded_file = st.sidebar.file_uploader("Or Upload Local Video File", type=["mp4", "mov", "avi"])

# Style selection checkboxes
st.sidebar.markdown("### 🎭 Caption Styles")
use_formal = st.sidebar.checkbox("Formal", value=True)
use_sarcastic = st.sidebar.checkbox("Sarcastic", value=True)
use_humorous_tech = st.sidebar.checkbox("Humorous Tech", value=True)
use_humorous_non_tech = st.sidebar.checkbox("Humorous Non-Tech", value=True)

# Allow inputting a custom style name
custom_style = st.sidebar.text_input("Add Custom Style (Optional)", placeholder="e.g. Shakespearean")

# Collect active styles list
active_styles = []
if use_formal: active_styles.append("formal")
if use_sarcastic: active_styles.append("sarcastic")
if use_humorous_tech: active_styles.append("humorous_tech")
if use_humorous_non_tech: active_styles.append("humorous_non_tech")
if custom_style.strip(): active_styles.append(custom_style.strip().lower().replace(" ", "_"))

# VLM API settings check
st.sidebar.markdown("### ⚙️ API Settings")
api_key = st.sidebar.text_input("Fireworks API Key", value=config.fireworks_api_key or "", type="password")

if api_key:
    # Dynamically apply the API key to configuration
    config.fireworks_api_key = api_key
    config.vlm_provider = "fireworks"
else:
    # If no API key is provided, we can offer running in mock mode for quick UI evaluation
    st.sidebar.warning("No Fireworks API Key provided. Running in **Mock Demo Mode**.")
    config.vlm_provider = "mock"

# 4. Main Panel Layout
st.markdown('<div class="main-title">CineScribe AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next-Generation Style-Conditioned Video Intelligence and Captioning Agent</div>', unsafe_allow_html=True)

# Main action button
run_clicked = st.button("🚀 Analyze & Generate Captions", type="primary", disabled=st.session_state["processing"])

if run_clicked:
    # Validate inputs
    final_video_source = None
    if uploaded_file is not None:
        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        with st.spinner("Saving uploaded file locally..."):
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        final_video_source = str(temp_file_path.resolve())
    elif video_url.strip():
        final_video_source = video_url.strip()
    
    if not final_video_source:
        st.error("Please provide a video source (select a preset, enter a custom URL, or upload a local file).")
    elif not active_styles:
        st.error("Please select at least one caption style.")
    else:
        st.session_state["processing"] = True
        st.session_state["result"] = None
        st.session_state["logs"] = []
        
        # UI log container setup
        st.markdown('<div class="section-header">⚙️ Real-time Pipeline Execution Logs</div>', unsafe_allow_html=True)
        log_placeholder = st.empty()
        
        # Async runner wrapped inside streamlit
        async def main_runner():
            handler = StreamlitLogHandler()
            app_logger.addHandler(handler)
            
            # Periodically poll and refresh the log placeholder
            async def log_refresher():
                while True:
                    if st.session_state["logs"]:
                        log_placeholder.code("\n".join(st.session_state["logs"][-15:]))
                    await asyncio.sleep(0.3)
                    
            refresher_task = asyncio.create_task(log_refresher())
            
            try:
                task = {
                    "task_id": "demo-session",
                    "video_url": final_video_source,
                    "styles": active_styles
                }
                # Run the backend execution manager
                results = await process_videos([], max_parallel=1, tasks=[task])
                return results[0]
            finally:
                refresher_task.cancel()
                app_logger.removeHandler(handler)

        try:
            with st.spinner("Running Video Intelligence Agent (Scene Extraction, Audio Transcription, VLM Analysis & Alignment)..."):
                result = asyncio.run(main_runner())
                
            if result.status == "failed":
                st.error("Pipeline execution failed. Review the logs above for stage details.")
                if result.stage_errors:
                    st.json(result.stage_errors)
            else:
                st.success("Successfully processed video and aligned styles!")
                st.session_state["result"] = result
        except Exception as ex:
            st.exception(ex)
        finally:
            st.session_state["processing"] = False

# 5. Render Results Dashboard
if st.session_state["result"]:
    res = st.session_state["result"]
    
    st.markdown('<div class="section-header">📊 Video Intelligence Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📺 Video Player")
        # Check if the source is a local path or a remote URL
        if os.path.exists(res.source):
            # Render local file upload
            with open(res.source, "rb") as vf:
                st.video(vf.read())
        else:
            # Render remote URL
            st.video(res.source)
            
        # Display tags
        if res.tags:
            st.markdown("#### 🏷️ Semantic Tags")
            tag_html = " ".join([f'<span style="background-color: #383a40; color: #FFFFFF; padding: 4px 10px; border-radius: 12px; margin-right: 5px; font-size: 0.85rem; border: 1px solid #4e5058;">{t}</span>' for t in res.tags])
            st.markdown(tag_html, unsafe_allow_html=True)
            
        # Performance metric cards
        st.markdown("#### ⚡ Pipeline Execution Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            total_time_s = sum(res.timings_ms.values()) / 1000.0
            st.markdown(f'<div class="metric-badge"><div class="metric-value">{total_time_s:.2f}s</div><div class="metric-label">Total Time</div></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="metric-badge"><div class="metric-value">{res.duration:.1f}s</div><div class="metric-label">Video Duration</div></div>', unsafe_allow_html=True)
        with m_col3:
            vlm_stages = res.timings_ms.get("vlm", 0) + res.timings_ms.get("summarize", 0)
            st.markdown(f'<div class="metric-badge"><div class="metric-value">{vlm_stages / 1000.0:.2f}s</div><div class="metric-label">LLM / VLM Overhead</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎭 Style-Conditioned Captions")
        
        # Iterate over and show the styled captions
        if res.captions:
            for style, text in res.captions.items():
                nice_name = style.replace("_", " ").title()
                header_class = f"style-{style}" if style in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"] else "style-custom"
                
                icon = "👔" if style == "formal" else "😏" if style == "sarcastic" else "💻" if style == "humorous_tech" else "😄" if style == "humorous_non_tech" else "✨"
                
                st.markdown(f"""
                <div class="caption-card">
                    <div class="style-header {header_class}">{icon} {nice_name}</div>
                    <div class="caption-content">{text}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No captions generated for selected styles.")
            
        # Add a download results button
        result_json_str = json.dumps({
            "task_id": res.id,
            "captions": res.captions or {},
            "summary": res.summary,
            "detailed_summary": res.detailed_summary,
            "tags": res.tags
        }, indent=2)
        st.download_button(
            label="💾 Download Captions JSON",
            data=result_json_str,
            file_name=f"cinescribe_{res.id}_results.json",
            mime="application/json",
            use_container_width=True
        )

    # Tabs for additional details (Timeline, Transcript, Technical metadata)
    st.markdown('<div class="section-header">🔍 Deep Analysis Timeline</div>', unsafe_allow_html=True)
    
    t_timeline, t_transcript, t_technical = st.tabs(["🕒 Event-Aligned Timeline", "🎙️ Audio Transcript", "📋 Raw Results JSON"])
    
    with t_timeline:
        if res.timeline:
            st.markdown("Here is the structured visual timeline merged by our agent from frame-level predictions:")
            for event in res.timeline:
                # Format start and end timestamps nicely
                time_range = f"{event.start:.1f}s - {event.end:.1f}s"
                st.markdown(f"""
                <div class="timeline-item">
                    <div class="timeline-time">{time_range}</div>
                    <div class="timeline-desc">{event.description}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No visual timeline events found. The video might not contain clear scene changes.")
            
    with t_transcript:
        if res.transcript:
            st.markdown(f"**Speech-to-Text Transcription:**")
            st.info(res.transcript)
        else:
            st.info("No audio speech detected or transcribed for this video.")
            
    with t_technical:
        st.markdown("Raw data structure exported by CineScribe schema:")
        st.json(res.model_dump())
