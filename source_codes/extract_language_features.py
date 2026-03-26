import os
import pandas as pd
import speech_recognition as sr
from textblob import TextBlob
import time
from pathlib import Path
from tqdm import tqdm

class LanguageFeatureExtractor:
    """Extract 5 language features from speech audio"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def transcribe_audio(self, audio_path):
        """Convert speech to text using Google's free Web Speech API"""
        try:
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise if necessary
                # self.recognizer.adjust_for_ambient_noise(source)
                audio_data = self.recognizer.record(source)
                
            # Using Google Web Speech API (requires internet, but free/no setup)
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "" # Could not understand audio
        except sr.RequestError as e:
            print(f"⚠️ API runtime error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Error processing {audio_path}: {e}")
            return None

    def extract_features(self, text):
        """Extract 5 text-based features from the transcription"""
        features = {
            'transcription': text,
            'word_count': 0,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'vocab_richness': 0.0,
            'sentence_complexity': 0.0,
        }
        
        if not text or len(text.strip()) == 0:
            return features
            
        blob = TextBlob(text)
        
        # 1. Word Count - Basic volume of speech
        words = blob.words
        word_count = len(words)
        features['word_count'] = word_count
        
        # 2. Sentiment Polarity (-1.0 to 1.0)
        # 3. Sentiment Subjectivity (0.0 to 1.0)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # 4. Vocabulary Richness (Type-Token Ratio)
        if word_count > 0:
            unique_words = set([w.lower() for w in words])
            features['vocab_richness'] = len(unique_words) / word_count
            
        # 5. Sentence Complexity (Average words per sentence)
        sentences = blob.sentences
        if len(sentences) > 0:
            features['sentence_complexity'] = word_count / len(sentences)
            
        return features
        
    def process_file(self, audio_path):
        """Transcribe and extract features for a single file"""
        text = self.transcribe_audio(audio_path)
        if text is None:
            return None
            
        features = self.extract_features(text)
        return features

    def process_batch(self, file_list, output_csv, batch_size=10):
        """Process multiple files"""
        results = []
        
        for i, filepath in enumerate(tqdm(file_list, desc="Extracting language features")):
            features = self.process_file(filepath)
            if features:
                features['filepath'] = str(filepath)
                features['filename'] = Path(filepath).name
                results.append(features)
                
            # Save intermediate results
            if (i + 1) % batch_size == 0:
                pd.DataFrame(results).to_csv(output_csv, index=False)
                
        # Final save
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(df)} features to {output_csv}")
        return df

if __name__ == "__main__":
    print("="*60)
    print("LANGUAGE FEATURE EXTRACTION PIPELINE")
    print("="*60)
    
    metadata_path = "dataset/cremad/crema_metadata.csv"
    if not Path(metadata_path).exists():
        print("❌ Metadata CSV not found. Run create_metadata.py first.")
        exit()
        
    metadata = pd.read_csv(metadata_path)
    
    # For testing, we won't process all 7441 files as API rate limits will hit
    print("⚠️ Testing Language Agent on a sample of 20 files to avoid API rate limits.")
    
    # Take a balanced sample: 10 happy, 10 sad
    happy_sample = metadata[metadata['emotion'] == 'happy'].head(10)
    sad_sample = metadata[metadata['emotion'] == 'sad'].head(10)
    test_sample = pd.concat([happy_sample, sad_sample])
    
    test_files = test_sample['path'].tolist()
    
    extractor = LanguageFeatureExtractor()
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_csv = "output/sample_language_features.csv"
    print("\n🚀 Starting transcription and extraction...")
    
    df = extractor.process_batch(test_files, output_csv)
    
    if len(df) > 0:
        print("\n📈 Sample Results:")
        print(df[['filename', 'transcription', 'sentiment_polarity']].head())
