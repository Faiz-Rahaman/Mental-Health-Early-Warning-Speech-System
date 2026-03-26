import streamlit as st
import sqlite3
import pandas as pd
import json
import os
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv
import uuid
import plotly.graph_objects as go
import plotly.express as px


# Import our LangGraph workflow
from langgraph_workflow import build_ewss_graph

load_dotenv()

# Page config
st.set_page_config(page_title="MH-EWSS Dashboard", page_icon="🎙️", layout="wide")

st.title("Mental Health Early Warning Speech System")
st.markdown("---")

DB_PATH = "output/ewss_features.db"

def fetch_latest_record():
    if not Path(DB_PATH).exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM feature_records ORDER BY timestamp DESC LIMIT 1", conn)
        return df.iloc[0] if not df.empty else None
    finally:
        conn.close()

def fetch_baseline():
    if not Path(DB_PATH).exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM personal_baselines WHERE user_id = 'user1'", conn)
        return df.iloc[0] if not df.empty else None
    finally:
        conn.close()

def derive_mental_health_status(risk_level):
    """Maps risk_level to mental health status"""
    if risk_level == 'High':
        return 'Unhealthy'
    elif risk_level == 'Medium':
        return 'Needs Attention'
    return 'Healthy'

def get_risk_recommendations(risk_level):
    """Returns specific recommendations based on risk level"""
    if risk_level == 'High':
        return [
            'Consider speaking with a trusted friend, family member, or counselor about how you are feeling.',
            'Try deep breathing exercises: inhale for 4 seconds, hold for 4 seconds, exhale for 6 seconds.',
            'Engage in light physical activity such as a 15-minute walk to regulate stress hormones.',
            'If feelings of distress persist, please reach out to a mental health helpline in your area.'
        ]
    elif risk_level == 'Medium':
        return [
            'Take a short break from your current activity and practice mindfulness for 5 minutes.',
            'Stay hydrated and ensure you have eaten a nutritious meal today.',
            'Consider journaling your thoughts to process any underlying stress or frustration.'
        ]
    return [
        'Your vocal patterns are within your normal range. Keep up your healthy routines!',
        'Continue maintaining regular sleep, exercise, and social connection habits.'
    ]

def run_ml_model(record):
    """Runs the trained ML model on the current recording features"""
    try:
        import joblib
        if os.path.exists("output/models/acoustic_risk_model.pkl"):
            md = joblib.load("output/models/acoustic_risk_model.pkl")
            ml_model = md['model']
            feat_names = md['features']
            class_map = md.get('classes', {0: 'Low', 1: 'Medium', 2: 'High'})
            fv = [record.get(f, 0.0) for f in feat_names]
            pc = int(ml_model.predict([fv])[0])
            ml_risk = class_map.get(pc, 'Unknown')
            probas = ml_model.predict_proba([fv])[0]
            confidence = float(max(probas)) * 100
            dataset = md.get('dataset', 'Unknown')
            return {
                'risk': ml_risk,
                'confidence': confidence,
                'probas': probas,
                'dataset': dataset,
                'model_data': md
            }
    except Exception:
        pass
    return None

def get_analysis_state(record, baseline):
    if record is None or baseline is None or record.empty or baseline.empty:
         return {"risk_level": "Unknown", "mental_health_status": "Unknown", "explanation": "Not enough data.", "recommendations": []}
    
    # Step 1: Run ML Model prediction FIRST
    ml_result = run_ml_model(record)
    ml_risk = ml_result['risk'] if ml_result else None
    ml_confidence = ml_result['confidence'] if ml_result else 0
    
    # Step 2: Get default recommendations from ML risk level
    default_recs = get_risk_recommendations(ml_risk or 'Low')
         
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        risk = ml_risk if ml_risk else "Low"
        return {
            "risk_level": risk,
            "mental_health_status": derive_mental_health_status(risk),
            "explanation": f"ML Model classified this audio as {risk} Risk with {ml_confidence:.1f}% confidence. (Set GEMINI_API_KEY for detailed AI explanation)",
            "recommendations": default_recs
        }
        
    # Safely extract values, defaulting None to 0
    pitch = record.get('pitch_mean') or 0
    speech_rate = record.get('speech_rate') or 0
    sentiment = record.get('sentiment_polarity') or 0
    bl_pitch = baseline.get('baseline_pitch_mean') or 0
    bl_speech = baseline.get('baseline_speech_rate') or 0
    bl_sentiment = baseline.get('baseline_sentiment') or 0
    
    prompt = "SYSTEM: You are a non-medical analysis agent. Review acoustic and language patterns. Do NOT diagnose medical conditions.\n\n"
    prompt += "--- CURRENT ---\n"
    prompt += f"Pitch: {pitch:.1f} Hz\n"
    prompt += f"Speech Rate: {speech_rate:.2f}\n"
    prompt += f"Sentiment: {sentiment:.2f}\n\n"
    
    prompt += "--- BASELINE ---\n"
    prompt += f"Average Pitch: {bl_pitch:.1f} Hz\n"
    prompt += f"Average Speech Rate: {bl_speech:.2f}\n"
    prompt += f"Average Sentiment: {bl_sentiment:.2f}\n\n"
    
    if ml_risk:
        prompt += "--- ML MODEL PREDICTION ---\n"
        prompt += f"GradientBoosting classifier (trained on DAIC-WOZ clinical data) predicts: {ml_risk} Risk (Confidence: {ml_confidence:.1f}%)\n"
        prompt += "Your risk_level MUST match the ML Model prediction unless linguistic sentiment STRONGLY contradicts it.\n\n"
    
    prompt += 'Provide JSON response: {"risk_level": "Low|Medium|High", "mental_health_status": "Healthy|Needs Attention|Unhealthy", "explanation": "Brief text", "recommendations": ["tip 1", "tip 2", "tip 3"]}'
    
    try:
        client = genai.Client(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            text = response.text.replace('```json', '').replace('```', '')
            try:
                result = json.loads(text)
            except:
                result = {"risk_level": ml_risk or "Error", "explanation": "Failed to parse LLM Response."}
        
        risk = result.get("risk_level", ml_risk or "Unknown")
        mhs = result.get("mental_health_status", derive_mental_health_status(risk))
        
        return {
            "risk_level": risk,
            "mental_health_status": mhs,
            "explanation": result.get("explanation", ""),
            "recommendations": result.get("recommendations", default_recs)
        }
    except Exception as e:
        risk = ml_risk or "Error"
        return {
            "risk_level": risk,
            "mental_health_status": derive_mental_health_status(risk),
            "explanation": f"Gemini API Error: {e}. ML Model predicted: {ml_risk} Risk.",
            "recommendations": default_recs
        }

def create_radar_chart(current, baseline):
    """Creates a Plotly radar chart comparing current features to baseline"""
    if baseline is None or baseline.empty:
        return None
        
    categories = ['Pitch', 'Speech Rate', 'Harshness (ZCR)', 'Brightness (Centroid)']
    
    # Normalize values for radar chart (simplistic normalization against baseline)
    cp = current.get('pitch_mean') or 0
    cs = current.get('speech_rate') or 0
    cz = current.get('zcr_mean') or 0
    cc = current.get('spectral_centroid_mean') or 0
    bp = baseline.get('baseline_pitch_mean') or 1
    bs = baseline.get('baseline_speech_rate') or 1
    bz = baseline.get('baseline_zcr') or 1
    bc = baseline.get('baseline_centroid') or 1
    
    current_vals = [
        cp / bp if bp > 0 else 1,
        cs / bs if bs > 0 else 1,
        cz / bz if bz > 0 else 1,
        cc / bc if bc > 0 else 1,
    ]
    
    # Baseline is exactly 1.0 everywhere after normalization
    baseline_vals = [1.0, 1.0, 1.0, 1.0]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=baseline_vals,
        theta=categories,
        fill='toself',
        name='Normal Baseline',
        line_color='rgba(169, 169, 169, 0.5)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=current_vals,
        theta=categories,
        fill='toself',
        name='Current Recording',
        line_color='rgba(30, 144, 255, 0.8)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(max(current_vals), 1.5)]),
        ),
        showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig


# Dashboard Layout
col_left, col_right = st.columns([1, 2], gap="large")

# Initialize the pipeline
@st.cache_resource
def get_pipeline():
    return build_ewss_graph()

pipeline = get_pipeline()

latest = fetch_latest_record()
base = fetch_baseline()

from database_manager import DatabaseManager

# Check if user already uploaded today
db = DatabaseManager(DB_PATH)
already_uploaded = db.has_uploaded_today("user1")

with col_left:
    st.header("📥 Audio Input")
    st.markdown("Upload your daily audio journal to track your vocal patterns over time.")
    
    if already_uploaded:
        st.info("🎯 You have already completed your daily audio check-in today! Please come back tomorrow to log another journal entry.")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Choose a .wav file", type=['wav'], label_visibility="collapsed")

    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Recording", type="primary", width='stretch'):
            with st.spinner("Processing audio through pipeline..."):
                temp_dir = Path("output/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / f"upload_{uuid.uuid4().hex[:8]}.wav"
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                initial_state = {"audio_path": str(temp_path), "user_id": "user1"}
                final_state = pipeline.invoke(initial_state)
                
                if final_state.get("error"):
                    st.error(f"Pipeline Error: {final_state['error']}")
                else:
                    st.toast("Analysis Complete!", icon="✅")
                    st.rerun()
        
    st.divider()
    
    st.header("📊 Your Baseline")
    if base is not None:
        st.success(f"Baseline active! Established over {base.get('total_samples', 0)} samples.")
        st.metric(label="Normal Pitch", value=f"{base.get('baseline_pitch_mean', 0):.1f} Hz")
        st.metric(label="Normal Speech Rate", value=f"{base.get('baseline_speech_rate', 0):.2f} syl/s")
        st.metric(label="Normal ZCR (Harshness)", value=f"{base.get('baseline_zcr', 0):.4f}")
    else:
        st.warning("Needs 5+ recordings to establish baseline. Keep uploading!")

with col_right:
    st.header("🧠 AI Analysis Results")
    if latest is not None:
        analysis = get_analysis_state(latest, base)
        
        # Also get ML model info for the probability display
        ml_result = run_ml_model(latest)
        
        # Mental Health Status Badge
        mental_status = analysis.get('mental_health_status', derive_mental_health_status(analysis.get('risk_level', 'Low')))
        if mental_status == 'Unhealthy':
            st.error("🧠 **Mental Health Status: Unhealthy** — Vocal biomarkers indicate signs of distress", icon="🔴")
        elif mental_status == 'Needs Attention':
            st.warning("🧠 **Mental Health Status: Needs Attention** — Some indicators suggest mild stress", icon="🟡")
        else:
            st.success("🧠 **Mental Health Status: Healthy** — Vocal patterns are within normal range", icon="🟢")
        
        # Risk Level Box
        if analysis['risk_level'] == 'Low':
            st.success(f"**Risk Level:** {analysis['risk_level']}", icon="🟢")
        elif analysis['risk_level'] == 'Medium':
            st.warning(f"**Risk Level:** {analysis['risk_level']}", icon="🟡")
        else:
            st.error(f"**Risk Level:** {analysis['risk_level']}", icon="🔴")
        
        # Show ML Model Prediction
        if ml_result:
            dataset_name = ml_result.get('dataset', 'Unknown')
            st.markdown(f"🤖 **ML Model Prediction:** `{ml_result['risk']} Risk` (Confidence: `{ml_result['confidence']:.1f}%`) — Trained on **{dataset_name}** dataset")
            ml_c1, ml_c2, ml_c3 = st.columns(3)
            ml_c1.metric("🟢 Low Risk", f"{ml_result['probas'][0]*100:.1f}%")
            ml_c2.metric("🟡 Medium Risk", f"{ml_result['probas'][1]*100:.1f}%")
            ml_c3.metric("🔴 High Risk", f"{ml_result['probas'][2]*100:.1f}%")
            
        with st.expander("Detailed AI Explanation", expanded=True):
            st.write(analysis['explanation'])
        
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(analysis['recommendations'], 1):
            st.markdown(f"**{i}.** {rec}")
        
        # Show specific guidance based on risk level
        if analysis['risk_level'] == 'High':
            st.error("⚠️ **If you are in crisis, please contact a mental health helpline immediately.**")
        elif analysis['risk_level'] == 'Medium':
            st.warning("💡 **Consider taking a break and practicing self-care today.**")
            
        st.divider()
        
        # Professional UI Layout for metrics
        st.subheader("🎙️ Voice Diagnostics")
        
        # Show Radar Chart if baseline exists
        radar_fig = create_radar_chart(latest, base)
        if radar_fig:
            st.plotly_chart(radar_fig, width='stretch')
        
        # Raw Metrics Grid — use (x or 0) because .get default won't catch explicit None values
        c1, c2, c3, c4 = st.columns(4)
        lp = latest.get('pitch_mean') or 0
        ls = latest.get('speech_rate') or 0
        lz = latest.get('zcr_mean') or 0
        lsent = latest.get('sentiment_polarity') or 0
        bp = (base.get('baseline_pitch_mean') or 0) if base is not None and not base.empty else 0
        bs = (base.get('baseline_speech_rate') or 0) if base is not None and not base.empty else 0
        bz = (base.get('baseline_zcr') or 0) if base is not None and not base.empty else 0
        bsent = (base.get('baseline_sentiment') or 0) if base is not None and not base.empty else 0
        
        c1.metric("Pitch", f"{lp:.1f} Hz", delta=f"{lp - bp:.1f}" if base is not None and not base.empty else None)
        c2.metric("Speech Rate", f"{ls:.2f}", delta=f"{ls - bs:.2f}" if base is not None and not base.empty else None)
        c3.metric("ZCR", f"{lz:.4f}", delta=f"{lz - bz:.4f}" if base is not None and not base.empty else None)
        c4.metric("Sentiment", f"{lsent:.2f}", delta=f"{lsent - bsent:.2f}" if base is not None and not base.empty else None)

    else:
        st.info("No recordings found in the database. Please upload an audio file to begin your analysis history.")

# Instructions at the bottom
st.markdown("---")

with st.expander("🔬 List of Extracted Features analyzed by the AI"):
    st.markdown("""
    **The Prosody Agent (Acoustic Physics)**
    
    - **1. Pitch Mean:** Average speaking frequency (Hz).
    - **2. Pitch Variability:** Standard deviation of pitch (monotone vs expressive).
    - **3. Energy Mean:** Overall loudness/volume.
    - **4. Energy Variability:** Dynamic range of volume.
    - **5. Speech Rate:** Speed of speaking (syllables per second).
    - **6. Pause Ratio:** Percentage of time spent in silence.
    - **7. Harmonics-to-Noise Ratio (HNR):** Voice clarity vs hoarseness.
    - **8. Zero Crossing Rate (ZCR):** Signal noisiness/harshness.
    - **9. Spectral Centroid:** Brightness of the voice.
    - **10. Spectral Rolloff:** High-frequency energy dropoff.
    
    **The Language Agent (Linguistic Analysis)**
    
    - **11. Transcription:** Raw text of spoken words.
    - **12. Word Count:** Total volume of words.
    - **13. Sentiment Polarity:** Positive (+1.0) or Negative (-1.0) emotion.
    - **14. Sentiment Subjectivity:** Objective facts vs Personal feelings.
    - **15. Vocabulary Richness:** Diversity of word choices.
    - **16. Sentence Complexity:** Grammatical density and length.
    """)

with st.expander("🛠️ Developer Actions"):
    st.code('''
# 1. Open Terminal
# 4. Run Streamlit from the root directory:
streamlit run source_codes/app_dashboard.py
''')
