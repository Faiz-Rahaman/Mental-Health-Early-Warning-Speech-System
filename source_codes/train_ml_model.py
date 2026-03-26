import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import os

def train_custom_model():
    print("="*60)
    print("TRAINING OPTIMIZED ML MODEL FOR MH-EWSS")
    print("="*60)

    features_path = "output/all_features_crema.csv"
    
    if not os.path.exists(features_path):
        print(f"❌ Could not find {features_path}. Please run extract_features.py on your dataset first.")
        return

    print("📊 Loading dataset...")
    df = pd.read_csv(features_path)

    features_list = [
        'pitch_mean', 'pitch_std', 'energy_mean', 'energy_std', 
        'speech_rate', 'pause_ratio', 'hnr', 
        'zcr_mean', 'spectral_centroid_mean', 'spectral_rolloff_mean'
    ]
    
    available_features = [f for f in features_list if f in df.columns]
    df = df.dropna(subset=available_features)
    
    if df.empty:
        print("❌ Dataset is empty after dropping missing values.")
        return
        
    print(f"✅ Loaded {len(df)} samples with {len(available_features)} acoustic features.")

    # Extract emotion labels from CREMA-D filenames
    if 'emotion_label' in df.columns:
        y_raw = df['emotion_label']
    elif 'emotion' in df.columns:
         y_raw = df['emotion']
    elif 'filename' in df.columns:
        y_raw = df['filename'].apply(lambda x: str(x).split('_')[2] if len(str(x).split('_')) >= 3 else 'NEU')
    else:
        print("❌ Could not find emotion labels in the dataset.")
        return

    # Map CREMA-D emotions to 3 risk classes for Mental Health context
    # 0 = Low Risk (Happy, Neutral)
    # 1 = Medium Risk (Anger, Disgust — elevated arousal but not depressive)
    # 2 = High Risk (Sadness, Fear — strong indicators of distress)
    def map_to_risk(emotion_code):
        code = str(emotion_code).upper()
        if code in ['SAD', 'FEA']:
            return 2  # High Risk
        elif code in ['ANG', 'DIS']:
            return 1  # Medium Risk
        return 0  # Low Risk
        
    y = y_raw.apply(map_to_risk)
    X = df[available_features]
    
    print(f"🎯 Target Distribution:")
    print(f"   Low Risk (0):    {sum(y==0)}")
    print(f"   Medium Risk (1): {sum(y==1)}")
    print(f"   High Risk (2):   {sum(y==2)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build a Pipeline with StandardScaler + GradientBoosting for better accuracy
    print("\n🤖 Training Gradient Boosting Classifier with StandardScaler...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=10,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\n📈 Evaluating Model...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk']))

    # 5-Fold Cross Validation
    print("🔄 Running 5-Fold Cross Validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"✅ 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

    # Save the model
    model_data = {
        'model': pipeline,
        'features': available_features,
        'classes': {0: 'Low', 1: 'Medium', 2: 'High'},
        'accuracy': acc
    }
    
    output_dir = Path("output/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "acoustic_risk_model.pkl"
    
    joblib.dump(model_data, model_path)
    print(f"\n💾 Model successfully saved to {model_path}!")
    print(f"   Model Type: GradientBoostingClassifier")
    print(f"   Classes: Low (0), Medium (1), High (2)")
    print(f"   Accuracy: {acc*100:.2f}%")
    print("You can now run your Streamlit dashboard and LangGraph will use this model automatically.")

if __name__ == "__main__":
    train_custom_model()
