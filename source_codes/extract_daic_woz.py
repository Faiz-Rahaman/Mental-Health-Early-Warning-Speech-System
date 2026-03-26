"""
DAIC-WOZ Clinical Dataset Feature Extractor
============================================
Extracts acoustic and linguistic features from the DAIC-WOZ (Distress Analysis 
Interview Corpus) clinical depression interview dataset.

Strategy:
- Each participant's 10-15 minute interview is chunked into 5-second speech segments
- This converts 6 participants into hundreds of usable training samples
- Each chunk gets the full 10 prosody features + 6 language features
- Transcript sentiment per-chunk provides clinical context labels

Dataset: DAIC-WOZ (USC Institute for Creative Technologies)
Reference: Gratch et al., "The Distress Analysis Interview Corpus", 2014
"""

import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

CHUNK_DURATION = 5  # seconds per audio chunk
SR = 16000          # sample rate

def extract_prosody_from_chunk(y_chunk, sr=SR):
    """Extract 10 acoustic features from a single audio chunk"""
    try:
        if len(y_chunk) < sr * 0.5:  # skip chunks shorter than 0.5s
            return None
            
        # 1-2. Pitch (Fundamental Frequency)
        f0, voiced_flag, _ = librosa.pyin(y_chunk, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.nanmean(valid_f0)) if len(valid_f0) > 0 else 0.0
        pitch_std = float(np.nanstd(valid_f0)) if len(valid_f0) > 0 else 0.0
        
        # 3-4. Energy (RMS)
        rms = librosa.feature.rms(y=y_chunk)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        
        # 5. Speech Rate (estimated from onset detection)
        onsets = librosa.onset.onset_detect(y=y_chunk, sr=sr)
        duration = len(y_chunk) / sr
        speech_rate = float(len(onsets) / duration) if duration > 0 else 0.0
        
        # 6. Pause Ratio
        intervals = librosa.effects.split(y_chunk, top_db=25)
        speaking_time = sum([end - start for start, end in intervals]) / sr
        pause_ratio = float((duration - speaking_time) / duration) if duration > 0 else 0.0
        
        # 7. HNR (Harmonics-to-Noise Ratio)
        harmonic = librosa.effects.harmonic(y_chunk)
        noise = y_chunk - harmonic
        hnr = float(10 * np.log10(np.mean(harmonic**2) / (np.mean(noise**2) + 1e-10)))
        
        # 8. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y_chunk)[0]
        zcr_mean = float(np.mean(zcr))
        
        # 9. Spectral Centroid (Brightness)
        centroid = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)[0]
        spectral_centroid_mean = float(np.mean(centroid))
        
        # 10. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr)[0]
        spectral_rolloff_mean = float(np.mean(rolloff))
        
        # 11. MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        
        features = {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'speech_rate': speech_rate,
            'pause_ratio': pause_ratio,
            'hnr': hnr,
            'zcr_mean': zcr_mean,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_rolloff_mean': spectral_rolloff_mean
        }
        
        for i in range(13):
            features[f'mfcc_{i+1}'] = float(mfcc_means[i])
            
        return features
    except Exception as e:
        return None

def get_transcript_chunks(transcript_path, chunk_duration=CHUNK_DURATION, total_duration=None):
    """Map transcript text to time-aligned chunks"""
    try:
        df = pd.read_csv(transcript_path, sep='\t')
        participant = df[df['speaker'] == 'Participant'].copy()
        
        if participant.empty or total_duration is None:
            return {}
            
        chunk_texts = {}
        num_chunks = int(total_duration / chunk_duration)
        
        for i in range(num_chunks):
            start = i * chunk_duration
            end = start + chunk_duration
            
            # Find all participant utterances that fall within this chunk's time window
            mask = (participant['start_time'] >= start) & (participant['start_time'] < end)
            chunk_utterances = participant[mask]['value'].dropna().astype(str).tolist()
            
            if chunk_utterances:
                text = " ".join(chunk_utterances)
                blob = TextBlob(text)
                words = blob.words
                word_count = len(words)
                
                chunk_texts[i] = {
                    'transcription': text[:200],
                    'word_count': word_count,
                    'sentiment_polarity': blob.sentiment.polarity,
                    'sentiment_subjectivity': blob.sentiment.subjectivity,
                    'vocabulary_richness': len(set(w.lower() for w in words)) / max(word_count, 1),
                    'sentence_complexity': word_count / max(len(blob.sentences), 1)
                }
            else:
                # Silence or interviewer-only chunk
                chunk_texts[i] = {
                    'transcription': '',
                    'word_count': 0,
                    'sentiment_polarity': 0.0,
                    'sentiment_subjectivity': 0.0,
                    'vocabulary_richness': 0.0,
                    'sentence_complexity': 0.0
                }
        return chunk_texts
    except Exception as e:
        print(f"  ⚠️ Error processing transcript: {e}")
        return {}

def label_chunk(prosody_feats, lang_feats):
    """
    Clinical risk labeling based on established vocal biomarkers of depression.
    
    Research basis (Cummins et al., 2015 - "A review of depression and suicide risk 
    assessment using speech analysis"):
    - Low pitch variability (monotone) → indicator of psychomotor retardation
    - High pause ratio → cognitive slowing
    - Low energy → reduced motivation
    - Negative sentiment → depressive cognition
    - Low speech rate → psychomotor slowing
    
    Returns: 0 (Low Risk), 1 (Medium Risk), 2 (High Risk)
    """
    risk_score = 0
    
    # Acoustic indicators
    if prosody_feats.get('pitch_std', 999) < 15:       # Monotone voice
        risk_score += 2
    if prosody_feats.get('pause_ratio', 0) > 0.6:      # Excessive pausing
        risk_score += 2
    if prosody_feats.get('energy_mean', 999) < 0.01:   # Very quiet/low energy
        risk_score += 1
    if prosody_feats.get('speech_rate', 999) < 2.0:    # Very slow speech
        risk_score += 1
    if prosody_feats.get('hnr', 999) < 5:              # Breathy/hoarse voice
        risk_score += 1
    if prosody_feats.get('spectral_centroid_mean', 999) < 1500:  # Dull voice
        risk_score += 1
        
    # Linguistic indicators
    if lang_feats.get('sentiment_polarity', 0) < -0.1:  # Negative sentiment
        risk_score += 2
    if lang_feats.get('word_count', 999) < 3:           # Very few words (withdrawal)
        risk_score += 1
    
    # Classify
    if risk_score >= 5:
        return 2  # High Risk
    elif risk_score >= 2:
        return 1  # Medium Risk
    return 0      # Low Risk

def process_daic_woz_dataset(base_dir="."):
    """Process all DAIC-WOZ participant folders into chunked, labeled training data"""
    print("="*60)
    print("DAIC-WOZ CLINICAL FEATURE EXTRACTION")
    print("Audio Chunking + Multi-Modal Feature Pipeline")
    print("="*60)
    
    participant_folders = sorted([
        f for f in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, f)) and f.endswith('_P')
    ])
    
    if not participant_folders:
        print("❌ No DAIC-WOZ participant folders found.")
        return
        
    print(f"🔎 Found {len(participant_folders)} clinical participants: {', '.join(participant_folders)}")
    
    all_samples = []
    
    from joblib import Parallel, delayed

    def process_single_chunk(i, y_chunk, sr, transcript_chunks, pid):
        """Helper function for parallel processing of a single audio chunk"""
        prosody = extract_prosody_from_chunk(y_chunk, sr)
        if prosody is None:
            return None
            
        lang = transcript_chunks.get(i, {
            'transcription': '', 'word_count': 0,
            'sentiment_polarity': 0.0, 'sentiment_subjectivity': 0.0,
            'vocabulary_richness': 0.0, 'sentence_complexity': 0.0
        })
        
        risk_label = label_chunk(prosody, lang)
        
        return {
            'participant_id': pid,
            'chunk_id': i,
            'chunk_start_sec': i * CHUNK_DURATION,
            **prosody,
            **lang,
            'risk_label': risk_label
        }

    for pf in participant_folders:
        pid = pf.split('_')[0]
        folder = os.path.join(base_dir, pf)
        audio_path = os.path.join(folder, f"{pid}_AUDIO.wav")
        transcript_path = os.path.join(folder, f"{pid}_TRANSCRIPT.csv")
        
        if not os.path.exists(audio_path):
            print(f"  ⚠️ Skipping {pf}: No audio file found")
            continue
            
        print(f"\n🔄 Processing Participant {pid} (Parallel CPU Mode)...")
        print(f"   Loading audio ({os.path.getsize(audio_path) / 1024 / 1024:.1f} MB)...")
        
        # Load full audio
        y, sr = librosa.load(audio_path, sr=SR)
        total_duration = len(y) / sr
        num_chunks = int(total_duration / CHUNK_DURATION)
        
        print(f"   Duration: {total_duration/60:.1f} minutes → {num_chunks} chunks of {CHUNK_DURATION}s each")
        
        # Get transcript chunks
        transcript_chunks = {}
        if os.path.exists(transcript_path):
            transcript_chunks = get_transcript_chunks(transcript_path, CHUNK_DURATION, total_duration)
        
        # Process chunks in parallel using joblib
        results = Parallel(n_jobs=-1, batch_size=5)(
            delayed(process_single_chunk)(
                i, 
                y[i * CHUNK_DURATION * sr : (i + 1) * CHUNK_DURATION * sr], 
                sr, 
                transcript_chunks, 
                pid
            ) 
            for i in range(num_chunks)
        )
        
        # Filter out None results
        valid_samples = [s for s in results if s is not None]
        all_samples.extend(valid_samples)
        
        print(f"   ✅ Extracted {len(valid_samples)} valid chunks from Participant {pid}")
    
    # Save
    df = pd.DataFrame(all_samples)
    
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "daic_woz_features.csv"
    df.to_csv(out_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"   Total Participants: {len(participant_folders)}")
    print(f"   Total Training Samples: {len(df)}")
    print(f"   Features per sample: 10 acoustic + 6 linguistic = 16")
    print(f"   Label Distribution:")
    for label, count in df['risk_label'].value_counts().sort_index().items():
        label_name = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}[label]
        print(f"      {label_name}: {count} samples ({count/len(df)*100:.1f}%)")
    print(f"\n💾 Saved to: {out_path}")

if __name__ == "__main__":
    process_daic_woz_dataset()
