"""
STEP 1: Map all your audio files to a CSV with emotion labels.
Run this FIRST to organize your dataset.
"""

import os
import pandas as pd
from pathlib import Path

# === CONFIG ===
AUDIO_FOLDER = "dataset/cremad/AudioWAV"
OUTPUT_CSV = "dataset/cremad/crema_metadata.csv"

# === Parse all WAV files ===
def create_metadata():
    print("📂 Scanning audio folder...")
    wav_files = list(Path(AUDIO_FOLDER).glob("*.wav"))
    print(f"✅ Found {len(wav_files)} WAV files")
    
    data = []
    emotion_map = {
        'ANG': 'anger', 'DIS': 'disgust', 'FEA': 'fear',
        'HAP': 'happy', 'SAD': 'sad', 'NEU': 'neutral'
    }
    
    for filepath in wav_files:
        filename = filepath.name
        # Parse: 1001_DFA_HAP_XX.wav
        parts = filename.replace('.wav', '').split('_')
        
        if len(parts) >= 4:
            actor = parts[0]
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code, 'unknown')
            
            # Gender: Actors 1001-1011 = Male, 1012-1091 = Female
            gender = 'male' if 1001 <= int(actor) <= 1011 else 'female'
            
            data.append({
                'filename': filename,
                'path': str(filepath),
                'actor': actor,
                'emotion': emotion,
                'emotion_code': emotion_code,
                'gender': gender
            })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved metadata to {OUTPUT_CSV}")
    print(f"\n📊 Dataset summary:")
    print(df['emotion'].value_counts())
    return df

if __name__ == "__main__":
    df = create_metadata()
    print("\n✅ First 5 rows:")
    print(df.head())