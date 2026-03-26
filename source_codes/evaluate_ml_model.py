import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import joblib

def generate_academic_graphs():
    print("="*60)
    print("GENERATING ACADEMIC ML EVALUATION GRAPHS")
    print("="*60)

    features_path = "output/all_features_crema.csv"
    if not os.path.exists(features_path):
        print(f"❌ Could not find {features_path}. Please run extract_features.py first.")
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
    
    # Extract emotion labels
    if 'emotion_label' in df.columns:
        y_raw = df['emotion_label']
    elif 'emotion' in df.columns:
         y_raw = df['emotion']
    elif 'filename' in df.columns:
        y_raw = df['filename'].apply(lambda x: str(x).split('_')[2] if len(str(x).split('_')) >= 3 else 'NEU')
    else:
        print("❌ Could not find emotion labels.")
        return

    # Map to High Risk (1) / Low Risk (0)
    def map_to_risk(emotion_code):
        high_risk_codes = ['SAD', 'ANG', 'FEA', 'DIS']
        if str(emotion_code).upper() in high_risk_codes:
            return 1
        return 0
        
    y = y_raw.apply(map_to_risk)
    X = df[available_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🤖 Training Model and Running 5-Fold Cross Validation...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # 1. Provide K-Fold Cross Validation Score (Judges love this)
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"✅ 5-Fold Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    output_dir = Path("output/evaluation_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # GRAPH 1: FEATURE IMPORTANCE (What parameters matter most?)
    # ---------------------------------------------------------
    print("📊 Plotting Feature Importance...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Acoustic Feature Importance (Random Forest)", fontsize=16, pad=15)
    sns.barplot(x=importances[indices], y=np.array(available_features)[indices], palette="viridis")
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Acoustic Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # GRAPH 2: CONFUSION MATRIX
    # ---------------------------------------------------------
    print("📊 Plotting Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Risk (0)', 'High Risk (1)'],
                yticklabels=['Low Risk (0)', 'High Risk (1)'],
                annot_kws={"size": 14})
    plt.title('Confusion Matrix (High vs Low Risk Assessment)', fontsize=16, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # GRAPH 3: ROC CURVE (Receiver Operating Characteristic)
    # ---------------------------------------------------------
    print("📊 Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, pad=15)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()

    print(f"\n✅ All academic graphs successfully saved to {output_dir}/")
    print("You can easily copy and paste these PNG images into your presentation slides!")

if __name__ == "__main__":
    generate_academic_graphs()
