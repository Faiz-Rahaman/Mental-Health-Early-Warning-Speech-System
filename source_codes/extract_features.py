import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm  # for progress bar

class ProsodyFeatureExtractor:
    """Extract 10 speech prosody features"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        
    def extract_all(self, audio_path):
        """Extract ALL features from one audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Skip if too short (< 1 second)
            if len(y) < sr:
                return None
                
            features = {}
            
            # 1. PITCH MEAN - Average speaking pitch
            pitches = librosa.yin(y, fmin=80, fmax=400)
            features['pitch_mean'] = np.nanmean(pitches) if np.any(~np.isnan(pitches)) else 0
            
            # 2. PITCH VARIABILITY - How monotone vs varied
            features['pitch_std'] = np.nanstd(pitches) if np.any(~np.isnan(pitches)) else 0
            
            # 3. ENERGY/LOUDNESS - Volume level
            rms = librosa.feature.rms(y=y)
            features['energy_mean'] = float(np.mean(rms))
            
            # 4. ENERGY VARIABILITY - Volume dynamics
            features['energy_std'] = float(np.std(rms))
            
            # 5. SPEECH RATE - Syllables per second (approx)
            # Using onset detection as proxy
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            duration = len(y) / sr
            features['speech_rate'] = len(onsets) / duration if duration > 0 else 0
            
            # 6. PAUSE RATIO - Silence proportion
            non_silent = librosa.effects.split(y, top_db=30)
            speech_duration = sum([end - start for start, end in non_silent]) / sr
            features['pause_ratio'] = 1 - (speech_duration / duration) if duration > 0 else 0
            
            # 7. HARMONICS-TO-NOISE RATIO - Voice clarity/quality
            try:
                harmonic, percussive = librosa.effects.hpss(y)
                features['hnr'] = float(np.mean(harmonic) / (np.mean(percussive) + 1e-6))
            except:
                features['hnr'] = 0
                
            # 8. ZERO CROSSING RATE - Noisiness/Harshness of speech
            features['zcr_mean'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            # 9. SPECTRAL CENTROID - "Brightness" of sound
            features['spectral_centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            
            # 10. SPECTRAL ROLLOFF - High-frequency energy
            features['spectral_rolloff_mean'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            
            # 11. MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            for i in range(13):
                features[f'mfcc_{i+1}'] = float(mfcc_means[i])
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error processing {audio_path}: {e}")
            return None
    
    def extract_batch(self, file_list, output_csv):
        """Extract features from multiple files with progress bar"""
        results = []
        
        # Use tqdm for progress bar
        for i, filepath in enumerate(tqdm(file_list, desc="Extracting features")):
            features = self.extract_all(filepath)
            if features:
                features['filepath'] = str(filepath)
                features['filename'] = Path(filepath).name
                results.append(features)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(df)} features to {output_csv}")
        return df

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("="*60)
    print("FEATURE EXTRACTION PIPELINE (FULL DATASET)")
    print("="*60)
    
    # 1. Load metadata to get all file paths
    metadata_path = "dataset/cremad/crema_metadata.csv"
    if not Path(metadata_path).exists():
        print("❌ Metadata CSV not found. Run create_metadata.py first.")
        exit()
    
    metadata = pd.read_csv(metadata_path)
    print(f"📊 Loaded metadata for {len(metadata)} audio files")
    
    # 2. Get all file paths
    all_files = metadata['path'].tolist()
    print(f"🎯 Target: {len(all_files)} files")
    
    # 3. Initialize extractor
    extractor = ProsodyFeatureExtractor()
    
    # 4. Ask for confirmation (since it takes time)
    print("\n⚠️ This will process all 7,441 files and take 5-15 minutes.")
    response = input("🔄 Proceed? (yes/no): ")
    
    if response.lower() == 'yes':
        print("\n🚀 Starting full feature extraction...")
        start_time = time.time()
        
        full_output = "output/all_features_crema.csv"
        df = extractor.extract_batch(all_files, full_output)
        
        elapsed = time.time() - start_time
        print(f"\n✅ COMPLETE! Time: {elapsed/60:.1f} minutes")
        print(f"📁 Output saved to: {full_output}")
        print(f"📊 Shape: {df.shape[0]} files, {df.shape[1]} features")
        
        # Quick summary statistics
        print("\n📈 Quick stats:")
        print(df[['pitch_mean', 'energy_mean', 'speech_rate', 'pause_ratio']].describe())
        
    else:
        print("❌ Extraction cancelled.")