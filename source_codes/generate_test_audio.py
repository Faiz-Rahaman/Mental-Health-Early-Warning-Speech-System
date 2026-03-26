"""
Generate synthetic test audio files that simulate different mental health states.
These are purely for demonstration and testing purposes.
"""
import numpy as np
import wave
import struct
from pathlib import Path

SR = 16000  # Sample rate

def generate_tone(freq, duration, sr=SR, amplitude=0.3):
    """Generate a simple sine tone"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def generate_silence(duration, sr=SR):
    """Generate silence"""
    return np.zeros(int(sr * duration), dtype=np.float32)

def save_wav(filename, audio_data, sr=SR):
    """Save audio as 16-bit WAV"""
    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(filename), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())

def generate_depressed_audio(output_path):
    """
    Generates audio that mimics DEPRESSED speech patterns:
    - Very low, monotone pitch (~100 Hz, almost no variation)
    - Long pauses between "words" (high pause ratio)
    - Very low energy/volume
    - Slow speech rate
    
    This should trigger HIGH RISK / UNHEALTHY in the model.
    """
    print("🔴 Generating DEPRESSED speech pattern (High Risk)...")
    
    segments = []
    
    # Pattern: short low monotone bursts with very long silences
    for i in range(8):
        # Very low pitch, low amplitude (quiet, monotone voice)
        freq = 95 + np.random.uniform(-3, 3)  # Almost no pitch variation (~95 Hz)
        tone = generate_tone(freq, duration=0.4, amplitude=0.08)  # Very quiet, short
        segments.append(tone)
        
        # Long pause (depressed speech has lots of silence)
        pause_duration = np.random.uniform(1.5, 3.0)
        segments.append(generate_silence(pause_duration))
    
    audio = np.concatenate(segments)
    save_wav(output_path, audio)
    print(f"   Saved to: {output_path} ({len(audio)/SR:.1f}s)")

def generate_healthy_audio(output_path):
    """
    Generates audio that mimics HEALTHY speech patterns:
    - Normal pitch with good variation (~180 Hz with ups and downs)
    - Short natural pauses
    - Normal energy levels
    - Normal speech rate
    
    This should trigger LOW RISK / HEALTHY in the model.
    """
    print("🟢 Generating HEALTHY speech pattern (Low Risk)...")
    
    segments = []
    
    # Pattern: varied pitch, normal pauses, good energy
    for i in range(15):
        # Varied pitch (expressive, healthy speech)
        freq = 180 + np.random.uniform(-40, 40)  # Good pitch variation
        duration = np.random.uniform(0.3, 0.8)    # Normal word lengths
        tone = generate_tone(freq, duration=duration, amplitude=0.35)  # Normal volume
        segments.append(tone)
        
        # Short natural pause
        pause_duration = np.random.uniform(0.1, 0.4)
        segments.append(generate_silence(pause_duration))
    
    audio = np.concatenate(segments)
    save_wav(output_path, audio)
    print(f"   Saved to: {output_path} ({len(audio)/SR:.1f}s)")

def generate_stressed_audio(output_path):
    """
    Generates audio that mimics STRESSED/ANXIOUS speech patterns:
    - Higher pitch (~220 Hz, tense)
    - Fast speech rate
    - Irregular pauses
    - Higher energy
    
    This should trigger MEDIUM RISK / NEEDS ATTENTION in the model.
    """
    print("🟡 Generating STRESSED speech pattern (Medium Risk)...")
    
    segments = []
    
    for i in range(20):
        # Higher, tense pitch
        freq = 220 + np.random.uniform(-20, 30)
        duration = np.random.uniform(0.15, 0.35)  # Fast, rushed speech
        tone = generate_tone(freq, duration=duration, amplitude=0.5)  # Louder
        segments.append(tone)
        
        # Very short, irregular pauses
        pause_duration = np.random.uniform(0.05, 0.2)
        segments.append(generate_silence(pause_duration))
    
    audio = np.concatenate(segments)
    save_wav(output_path, audio)
    print(f"   Saved to: {output_path} ({len(audio)/SR:.1f}s)")

if __name__ == "__main__":
    print("="*60)
    print("GENERATING TEST AUDIO FILES FOR DEMO")
    print("="*60)
    
    output_dir = Path("output/test_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_depressed_audio(output_dir / "test_DEPRESSED_high_risk.wav")
    generate_stressed_audio(output_dir / "test_STRESSED_medium_risk.wav")
    generate_healthy_audio(output_dir / "test_HEALTHY_low_risk.wav")
    
    print(f"\n✅ All 3 test files saved to: {output_dir}/")
    print("Upload these to the Streamlit dashboard to test each risk level!")
