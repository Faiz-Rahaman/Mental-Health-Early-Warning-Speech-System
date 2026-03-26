"""
STEP 2: Test if you can load audio and extract basic features.
Run this AFTER creating metadata.
"""

import librosa
import pandas as pd
import numpy as np

# Load metadata
df = pd.read_csv(r"dataset\cremad\crema_metadata.csv")

def test_audio_loading():
    """Test 1: Can you load audio?"""
    print("🔊 Testing audio loading...")
    
    # Take first file
    sample = df.iloc[0]
    filepath = sample['path']
    
    # Load audio
    y, sr = librosa.load(filepath, sr=16000, duration=3.0)
    
    print(f"✅ Loaded: {sample['filename']}")
    print(f"   Duration: {len(y)/sr:.2f} seconds")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Audio shape: {y.shape}")
    
    return y, sr

def test_basic_features():
    """Test 2: Can you extract simple features?"""
    print("\n📊 Testing feature extraction...")
    
    # Compare happy vs sad for same actor
    actor = 1001  # Male actor
    happy_file = df[(df['actor']==actor) & (df['emotion']=='happy')].iloc[0]
    sad_file = df[(df['actor']==actor) & (df['emotion']=='sad')].iloc[0]
    
    features = {}
    
    for emotion, fileinfo in [('happy', happy_file), ('sad', sad_file)]:
        y, sr = librosa.load(fileinfo['path'], sr=16000)
        
        # 1. Pitch (F0) - fundamental frequency
        pitches = librosa.yin(y, fmin=80, fmax=400)
        pitch_mean = np.nanmean(pitches)
        
        # 2. Energy (RMS)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        
        features[emotion] = {
            'pitch': pitch_mean,
            'energy': energy_mean
        }
    
    print(f"🎵 Happy: Pitch={features['happy']['pitch']:.1f}Hz, Energy={features['happy']['energy']:.4f}")
    print(f"🎵 Sad:   Pitch={features['sad']['pitch']:.1f}Hz, Energy={features['sad']['energy']:.4f}")
    
    # Expected: Sad should have LOWER pitch
    pitch_diff = ((features['sad']['pitch'] - features['happy']['pitch']) / features['happy']['pitch']) * 100
    print(f"\n📈 Pitch difference: {pitch_diff:.1f}% {'(Expected: negative)' if pitch_diff < 0 else '(Unexpected: positive)'}")
    
    return features

if __name__ == "__main__":
    print("="*50)
    print("AUDIO PIPELINE TEST")
    print("="*50)
    
    test_audio_loading()
    test_basic_features()