import os
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Import the previously created standalone agents/extractors
from extract_features import ProsodyFeatureExtractor
from extract_language_features import LanguageFeatureExtractor
from database_manager import DatabaseManager

# 1. Define the Shared State for the Graph
class EWSSState(TypedDict):
    """The state passed between agents in the graph"""
    audio_path: str
    user_id: str
    prosody_features: Optional[Dict[str, Any]]
    language_features: Optional[Dict[str, Any]]
    baseline_data: Optional[Dict[str, Any]]
    ml_acoustic_prediction: Optional[float]
    analysis_result: Optional[Dict[str, Any]]
    error: Optional[str]

# 2. Define the Nodes (Agents)
def prosody_agent(state: EWSSState):
    """Node that handles acoustic processing"""
    print("🎙️ [Prosody Agent] Extracting acoustic features...")
    
    if not os.path.exists(state["audio_path"]):
        return {"error": f"Audio file not found: {state['audio_path']}"}
        
    extractor = ProsodyFeatureExtractor()
    features = extractor.extract_all(state["audio_path"])
    
    return {"prosody_features": features if features else {}}

def language_agent(state: EWSSState):
    """Node that handles speech-to-text and sentiment"""
    print("📝 [Language Agent] Extracting transcription and language features...")
    
    if state.get("error"):
        return {} # Pass through if error
        
    extractor = LanguageFeatureExtractor()
    features = extractor.process_file(state["audio_path"])
    
    return {"language_features": features if features else {}}

def temporal_agent(state: EWSSState):
    """Node that handles database storage and baseline comparisons"""
    print("⏳ [Temporal Agent] Fetching user baseline from database...")
    
    if state.get("error"):
        return {}
        
    db = DatabaseManager()
    
    # Store the recently extracted features
    db.insert_record(
        user_id=state["user_id"],
        audio_path=state["audio_path"],
        prosody_features=state.get("prosody_features", {}),
        language_features=state.get("language_features", {})
    )
    
    # Fetch/Update Baseline
    baseline = db.calculate_update_baseline(state["user_id"])
    return {"baseline_data": baseline}

def ml_specialist_agent(state: EWSSState):
    """Node that evaluates just the raw numbers using our custom trained ML model"""
    print("🤖 [ML Specialist Agent] Running custom acoustic ML model...")
    
    model_path = "output/models/acoustic_risk_model.pkl"
    
    if not os.path.exists(model_path) or state.get("error"):
        print("ℹ️ Custom ML model not found. Skipping ML predictions.")
        return {"ml_acoustic_prediction": None}
        
    try:
        import joblib
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_names = model_data['features']
        class_map = model_data.get('classes', {0: 'Low', 1: 'Medium', 2: 'High'})
        
        prosody = state.get("prosody_features", {})
        
        # Prepare the exact feature vector the model expects
        feature_vector = []
        for feat in feature_names:
            feature_vector.append(prosody.get(feat, 0.0))
            
        # Predict the risk class directly
        predicted_class = int(model.predict([feature_vector])[0])
        predicted_label = class_map.get(predicted_class, 'Unknown')
        
        # Get probability scores for all classes
        probas = model.predict_proba([feature_vector])[0]
        confidence = float(max(probas)) * 100
        
        print(f"✅ ML Model Prediction: {predicted_label} Risk (Confidence: {confidence:.1f}%)")
        print(f"   Probabilities -> Low: {probas[0]*100:.1f}%, Medium: {probas[1]*100:.1f}%, High: {probas[2]*100:.1f}%")
        
        return {"ml_acoustic_prediction": {
            'risk_level': predicted_label,
            'confidence': confidence,
            'probabilities': {'Low': float(probas[0]), 'Medium': float(probas[1]), 'High': float(probas[2])}
        }}
        
    except Exception as e:
        print(f"⚠️ Error running custom ML model: {e}")
        return {"ml_acoustic_prediction": None}

def llm_decision_agent(state: EWSSState):
    """Node that analyzes features against baseline using Gemini API to determine risk"""
    print("🧠 [LLM Decision Agent] Analyzing features against baseline...")
    
    if state.get("error"):
        return {"analysis_result": {"risk_level": "Error", "explanation": state["error"], "recommendations": []}}
        
    prosody = state.get("prosody_features", {})
    language = state.get("language_features", {})
    baseline = state.get("baseline_data")
    
    # Fallback to empty if None
    prosody = prosody if prosody is not None else {}
    language = language if language is not None else {}
    
    # Generate Prompt
    prompt = "SYSTEM: You are a non-medical analysis agent. Your task is to review acoustic and language patterns. Do NOT diagnose depression or any disorder.\\n\\n"
    
    prompt += "--- CURRENT RECORDING ---\\n"
    prompt += f"Transcription: '{language.get('transcription', 'N/A')}'\\n"
    prompt += f"Pitch: {prosody.get('pitch_mean', 0):.1f} Hz\\n"
    prompt += f"Speech Rate: {prosody.get('speech_rate', 0):.2f}\\n"
    prompt += f"Zero Crossing Rate (Harshness): {prosody.get('zcr_mean', 0):.4f}\\n"
    prompt += f"Spectral Centroid (Brightness): {prosody.get('spectral_centroid_mean', 0):.1f} Hz\\n"
    prompt += f"Sentiment Polarity: {language.get('sentiment_polarity', 0):.2f} (-1 to 1)\\n\\n"
    
    if baseline:
        prompt += "--- PERSONAL BASELINE (Normal State) ---\\n"
        prompt += f"Average Pitch: {baseline.get('baseline_pitch_mean', 0):.1f} Hz\\n"
        prompt += f"Average Speech Rate: {baseline.get('baseline_speech_rate', 0):.2f}\\n"
        prompt += f"Average Zero Crossing Rate: {baseline.get('baseline_zcr', 0):.4f}\\n"
        prompt += f"Average Spectral Centroid: {baseline.get('baseline_centroid', 0):.1f} Hz\\n"
        prompt += f"Average Sentiment: {baseline.get('baseline_sentiment', 0):.2f}\\n\\n"
        
        # Simple heuristic trend
        pitch_drop = prosody.get('pitch_mean', 0) < baseline.get('baseline_pitch_mean', 0) * 0.9
        slow_speech = prosody.get('speech_rate', 0) < baseline.get('baseline_speech_rate', 0) * 0.85
        zcr_drop = prosody.get('zcr_mean', 0) < baseline.get('baseline_zcr', 0) * 0.8
        prompt += f"TRENDS: Pitch Drop Detected={pitch_drop}, Slower Speech Detected={slow_speech}, Energy/Brightness Drop Detected={zcr_drop}\\n\\n"
    
    # Integrate the Custom ML Model prediction if available
    ml_pred = state.get("ml_acoustic_prediction")
    if ml_pred is not None and isinstance(ml_pred, dict):
        ml_risk = ml_pred.get('risk_level', 'Unknown')
        ml_conf = ml_pred.get('confidence', 0)
        ml_probs = ml_pred.get('probabilities', {})
        prompt += "--- DEDICATED ACOUSTIC ML MODEL PREDICTION ---\\n"
        prompt += f"Our custom trained GradientBoosting classifier (trained on DAIC-WOZ clinical interview samples) analyzed the raw physics of this audio signal.\\n"
        prompt += f"ML Model Prediction: {ml_risk} Risk (Confidence: {ml_conf:.1f}%)\\n"
        prompt += f"Probability Breakdown -> Low: {ml_probs.get('Low', 0)*100:.1f}%, Medium: {ml_probs.get('Medium', 0)*100:.1f}%, High: {ml_probs.get('High', 0)*100:.1f}%\\n"
        prompt += "IMPORTANT: Your final risk_level MUST match the ML Model's prediction unless the linguistic sentiment strongly contradicts it.\\n\\n"
        
        # Add specific recommendations based on ML risk level
        if ml_risk == 'High':
            prompt += "--- REQUIRED RECOMMENDATIONS FOR HIGH RISK ---\\n"
            prompt += "You MUST include these recommendations:\\n"
            prompt += '1. "Consider speaking with a trusted friend, family member, or counselor about how you are feeling."\\n'
            prompt += '2. "Try deep breathing exercises: inhale for 4 seconds, hold for 4 seconds, exhale for 6 seconds."\\n'
            prompt += '3. "Engage in light physical activity such as a 15-minute walk to regulate stress hormones."\\n'
            prompt += '4. "If feelings of distress persist, please reach out to a mental health helpline in your area."\\n\\n'
        elif ml_risk == 'Medium':
            prompt += "--- REQUIRED RECOMMENDATIONS FOR MEDIUM RISK ---\\n"
            prompt += "You MUST include these recommendations:\\n"
            prompt += '1. "Take a short break from your current activity and practice mindfulness for 5 minutes."\\n'
            prompt += '2. "Stay hydrated and ensure you have eaten a nutritious meal today."\\n'
            prompt += '3. "Consider journaling your thoughts to process any underlying stress or frustration."\\n\\n'
        else:
            prompt += "--- REQUIRED RECOMMENDATIONS FOR LOW RISK ---\\n"
            prompt += "You MUST include these recommendations:\\n"
            prompt += '1. "Your vocal patterns are within your normal range. Keep up your healthy routines!"\\n'
            prompt += '2. "Continue maintaining regular sleep, exercise, and social connection habits."\\n\\n'
        
    prompt += "--- REQUIRED OUTPUT ---\\n"
    prompt += "Provide JSON formatted response with exactly these keys:\\n"
    prompt += '{"risk_level": "Low|Medium|High", "mental_health_status": "Healthy|Needs Attention|Unhealthy", "explanation": "Brief text explaining the analysis", "recommendations": ["tip 1", "tip 2", "tip 3"]}'
    
    # Call Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not set. Falling back to mock response.")
        # Use ML model prediction if available for mock response
        ml_pred_mock = state.get("ml_acoustic_prediction")
        if ml_pred_mock and isinstance(ml_pred_mock, dict):
            risk_level = ml_pred_mock.get('risk_level', 'Low')
            if risk_level == 'High':
                mental_health_status = 'Unhealthy'
                explanation = 'Your acoustic features suggest elevated vocal distress markers including reduced pitch variation and increased pause ratio. (MOCKED - Set GEMINI_API_KEY for detailed analysis)'
                recommendations = [
                    'Consider speaking with a trusted friend, family member, or counselor about how you are feeling.',
                    'Try deep breathing exercises: inhale for 4 seconds, hold for 4 seconds, exhale for 6 seconds.',
                    'Engage in light physical activity such as a 15-minute walk to regulate stress hormones.',
                    'If feelings of distress persist, please reach out to a mental health helpline in your area.'
                ]
            elif risk_level == 'Medium':
                mental_health_status = 'Needs Attention'
                explanation = 'Some acoustic indicators suggest mild stress or fatigue in your speech patterns. (MOCKED - Set GEMINI_API_KEY for detailed analysis)'
                recommendations = [
                    'Take a short break from your current activity and practice mindfulness for 5 minutes.',
                    'Stay hydrated and ensure you have eaten a nutritious meal today.',
                    'Consider journaling your thoughts to process any underlying stress or frustration.'
                ]
            else:
                mental_health_status = 'Healthy'
                explanation = 'Your vocal patterns appear stable and within healthy ranges. (MOCKED - Set GEMINI_API_KEY for detailed analysis)'
                recommendations = [
                    'Your vocal patterns are within your normal range. Keep up your healthy routines!',
                    'Continue maintaining regular sleep, exercise, and social connection habits.'
                ]
        else:
            mental_health_status = 'Healthy'
                    
        return {"analysis_result": {"risk_level": risk_level, "mental_health_status": mental_health_status, "explanation": explanation, "recommendations": recommendations}}
    
    print("🔄 Calling Gemini API...")
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
            print("⚠️ Failed to parse JSON from LLM. Raw text:", response.text)
            result = {"risk_level": "Unknown", "explanation": response.text, "recommendations": []}
            
        analysis = {
            "risk_level": result.get("risk_level", "Unknown"),
            "explanation": result.get("explanation", "Parsed response missing explanation."),
            "recommendations": result.get("recommendations", [])
        }
        print("✅ Analysis Complete via Gemini API.")
        
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        analysis = {
            "risk_level": "Error",
            "explanation": f"API call failed: {e}",
            "recommendations": []
        }
        
    return {"analysis_result": analysis}

# 3. Build and Compile the Graph
def build_ewss_graph():
    """Constructs the LangGraph computational graph"""
    workflow = StateGraph(EWSSState)
    
    # Add our agent nodes
    workflow.add_node("prosody", prosody_agent)
    workflow.add_node("language", language_agent)
    workflow.add_node("temporal", temporal_agent)
    workflow.add_node("ml_specialist", ml_specialist_agent)
    workflow.add_node("decision", llm_decision_agent)
    
    # Define execution order
    workflow.set_entry_point("prosody")
    workflow.add_edge("prosody", "language")
    workflow.add_edge("language", "temporal")
    workflow.add_edge("temporal", "ml_specialist")
    workflow.add_edge("ml_specialist", "decision")
    workflow.add_edge("decision", END)
    
    return workflow.compile()

if __name__ == "__main__":
    print("="*60)
    print("INITIALIZING LANGGRAPH WORKFLOW")
    print("="*60)
    
    app = build_ewss_graph()
    
    # For testing, grab the first file from the CREMA dataset
    test_audio = "dataset/cremad/AudioWAV/1001_DFA_ANG_XX.wav"
    
    if not os.path.exists(test_audio):
         print(f"⚠️ Test audio not found at {test_audio}. Cannot execute test run.")
    else:
        initial_state = {
            "audio_path": test_audio,
            "user_id": "user1"
        }
        
        print(f"\n🚀 Running Pipeline for: {test_audio}")
        final_state = app.invoke(initial_state)
        
        print("\n" + "="*60)
        print("🏁 FINAL SYSTEM OUTPUT (Dashboard View):")
        print("="*60)
        print(json.dumps(final_state.get("analysis_result", {}), indent=2))
