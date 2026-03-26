import sqlite3
import pandas as pd
from pathlib import Path
import os
import datetime

class DatabaseManager:
    """Manages the SQLite database for the EWSS Project."""
    
    def __init__(self, db_path="output/ewss_features.db"):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def _initialize_db(self):
        """Create the necessary tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Table for storing daily/individual recordings
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'user1',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            audio_path TEXT,
            
            -- Prosody (How)
            pitch_mean REAL,
            pitch_std REAL,
            energy_mean REAL,
            energy_std REAL,
            speech_rate REAL,
            pause_ratio REAL,
            hnr REAL,
            zcr_mean REAL,
            spectral_centroid_mean REAL,
            spectral_rolloff_mean REAL,
            
            -- Language (What)
            transcription TEXT,
            word_count INTEGER,
            sentiment_polarity REAL,
            sentiment_subjectivity REAL,
            vocab_richness REAL,
            sentence_complexity REAL,
            
            -- Metadata
            emotion_label TEXT
        )
        ''')
        
        # Table for storing the calculated personal baseline
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS personal_baselines (
            user_id TEXT PRIMARY KEY,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_samples INTEGER,
            
            baseline_pitch_mean REAL,
            baseline_energy_mean REAL,
            baseline_speech_rate REAL,
            baseline_sentiment REAL,
            baseline_zcr REAL,
            baseline_centroid REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")

    def insert_record(self, user_id, audio_path, prosody_features, language_features, emotion_label="unknown"):
        """Insert a new feature record into the database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO feature_records (
            user_id, audio_path, 
            pitch_mean, pitch_std, energy_mean, energy_std, speech_rate, pause_ratio, hnr,
            zcr_mean, spectral_centroid_mean, spectral_rolloff_mean,
            transcription, word_count, sentiment_polarity, sentiment_subjectivity, vocab_richness, sentence_complexity,
            emotion_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, str(audio_path),
            prosody_features.get('pitch_mean'), prosody_features.get('pitch_std'), 
            prosody_features.get('energy_mean'), prosody_features.get('energy_std'), 
            prosody_features.get('speech_rate'), prosody_features.get('pause_ratio'), 
            prosody_features.get('hnr'),
            prosody_features.get('zcr_mean'), prosody_features.get('spectral_centroid_mean'),
            prosody_features.get('spectral_rolloff_mean'),
            
            language_features.get('transcription'), language_features.get('word_count'),
            language_features.get('sentiment_polarity'), language_features.get('sentiment_subjectivity'),
            language_features.get('vocab_richness'), language_features.get('sentence_complexity'),
            
            emotion_label
        ))
        
        conn.commit()
        conn.close()
        print(f"💾 Inserted new record for {user_id}: {os.path.basename(audio_path)}")

    def get_all_records(self, user_id='user1'):
        """Retrieve all records for a user as a pandas DataFrame"""
        conn = self._get_connection()
        query = "SELECT * FROM feature_records WHERE user_id = ?"
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        return df
        
    def has_uploaded_today(self, user_id='user1'):
        """Check if the user has already uploaded a recording today (local server time)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        query = (
            "SELECT COUNT(*) FROM feature_records "
            "WHERE user_id = ? AND date(timestamp, 'localtime') = date('now', 'localtime')"
        )
        cursor.execute(query, (user_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
        
    def calculate_update_baseline(self, user_id='user1'):
        """Calculate the average baseline features for a user and save it"""
        df = self.get_all_records(user_id)
        
        if len(df) < 5:
            print(f"⚠️ Need at least 5 records to establish a baseline. Currently have {len(df)}.")
            return None
            
        baseline = {
            'user_id': user_id,
            'total_samples': len(df),
            'baseline_pitch_mean': df['pitch_mean'].mean(),
            'baseline_energy_mean': df['energy_mean'].mean(),
            'baseline_speech_rate': df['speech_rate'].mean(),
            'baseline_sentiment': df['sentiment_polarity'].mean(),
            'baseline_zcr': df['zcr_mean'].mean() if 'zcr_mean' in df.columns else 0.0,
            'baseline_centroid': df['spectral_centroid_mean'].mean() if 'spectral_centroid_mean' in df.columns else 0.0
        }
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # UPSERT logic (Insert or Replace)
        cursor.execute('''
        INSERT OR REPLACE INTO personal_baselines (
            user_id, updated_at, total_samples, 
            baseline_pitch_mean, baseline_energy_mean, baseline_speech_rate, baseline_sentiment,
            baseline_zcr, baseline_centroid
        ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            baseline['user_id'], baseline['total_samples'],
            baseline['baseline_pitch_mean'], baseline['baseline_energy_mean'],
            baseline['baseline_speech_rate'], baseline['baseline_sentiment'],
            baseline['baseline_zcr'], baseline['baseline_centroid']
        ))
        
        conn.commit()
        conn.close()
        print(f"📊 Baseline updated for {user_id} based on {len(df)} samples.")
        return baseline
        
    def mock_populate_database(self, prosody_csv, language_csv):
        """Helper function to load data from the CSVs created earlier for testing"""
        print("🔄 Migrating data from CSVs to SQLite...")
        
        if not os.path.exists(prosody_csv) or not os.path.exists(language_csv):
            print("❌ CSV files missing. Please run the extraction scripts first.")
            return
            
        p_df = pd.read_csv(prosody_csv)
        l_df = pd.read_csv(language_csv)
        
        # Merge on filename
        merged_df = pd.merge(p_df, l_df, on='filename', suffixes=('', '_lang'))
        
        for index, row in merged_df.iterrows():
            prosody_feats = {
                'pitch_mean': row['pitch_mean'], 'pitch_std': row['pitch_std'],
                'energy_mean': row['energy_mean'], 'energy_std': row['energy_std'],
                'speech_rate': row['speech_rate'], 'pause_ratio': row['pause_ratio'], 'hnr': row['hnr']
            }
            
            language_feats = {
                'transcription': row.get('transcription', ''), 'word_count': row.get('word_count', 0),
                'sentiment_polarity': row.get('sentiment_polarity', 0), 
                'sentiment_subjectivity': row.get('sentiment_subjectivity', 0),
                'vocab_richness': row.get('vocab_richness', 0), 'sentence_complexity': row.get('sentence_complexity', 0)
            }
            
            emotion = row.get('emotion', 'unknown')
            self.insert_record('user1', row['filepath'], prosody_feats, language_feats, emotion)
            
        print(f"✅ Successfully migrated {len(merged_df)} records.")
        self.calculate_update_baseline('user1')


if __name__ == "__main__":
    print("="*60)
    print("DATABASE MANAGER SETUP")
    print("="*60)
    db = DatabaseManager()
    
    # Try to populate if the user ran both extraction scripts
    p_csv = "output/all_features_crema.csv"
    l_csv = "output/sample_language_features.csv"
    
    if os.path.exists(p_csv) and os.path.exists(l_csv):
        db.mock_populate_database(p_csv, l_csv)
    else:
        print("ℹ️ Standard tables created. Ready for incoming data.")
