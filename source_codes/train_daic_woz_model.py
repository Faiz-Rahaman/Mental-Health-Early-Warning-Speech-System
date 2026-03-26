"""
DAIC-WOZ Clinical ML Model Training
=====================================
Trains a GradientBoosting classifier on the chunked DAIC-WOZ clinical data.
The model predicts 3-class risk levels (Low / Medium / High) and is saved
as the primary model used by the LangGraph ML Specialist Agent.

After training, this model REPLACES the CREMA-D model as the production model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def train_daic_woz_model():
    print("="*60)
    print("TRAINING CLINICAL ML MODEL ON DAIC-WOZ DATA")
    print("="*60)

    features_path = "output/daic_woz_features.csv"
    
    if not os.path.exists(features_path):
        print(f"❌ Could not find {features_path}. Please run extract_daic_woz.py first.")
        return

    df = pd.read_csv(features_path)
    print(f"📊 Loaded {len(df)} clinical audio samples from DAIC-WOZ.")
    
    # Features used by the model (must match what LangGraph feeds it)
    features_list = [
        'pitch_mean', 'pitch_std', 'energy_mean', 'energy_std', 
        'speech_rate', 'pause_ratio', 'hnr',
        'zcr_mean', 'spectral_centroid_mean', 'spectral_rolloff_mean'
    ]
    for i in range(1, 14):
        features_list.append(f'mfcc_{i}')
    
    available_features = [f for f in features_list if f in df.columns]
    print(f"✅ Using {len(available_features)} acoustic features: {available_features}")
    
    X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['risk_label']
    
    print(f"\n🎯 Label Distribution BEFORE SMOTE:")
    label_names = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    for label in sorted(y.unique()):
        count = sum(y == label)
        print(f"   {label_names.get(label, label)}: {count} samples ({count/len(y)*100:.1f}%)")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📐 Original Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Apply SMOTE mathematically clone data to balance classes
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"⚖️ Applied SMOTE! New Balanced Train: {len(X_train)} samples")
    except ImportError as e:
        print(f"⚠️ 'imbalanced-learn' import failed: {e}. Skipping SMOTE. Run: pip install imbalanced-learn")
    except Exception as e:
        print(f"⚠️ SMOTE balancing failed due to error: {e}")
    
    # Build Pipeline: StandardScaler + GradientBoosting
    print("\n🤖 Training GradientBoosting Classifier (with StandardScaler)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\n📈 Model Evaluation:")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n   Test Accuracy: {acc*100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk']))

    # Cross Validation
    print("🔄 Running 5-Fold Stratified Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"   5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   {'':>12} Pred Low  Pred Med  Pred High")
    for i, row_name in enumerate(['True Low', 'True Med', 'True High']):
        if i < len(cm):
            print(f"   {row_name:>12}  {cm[i][0]:>7}  {cm[i][1]:>8}  {cm[i][2]:>9}")

    # ----- SAVE CONFUSION MATRIX PLOT -----
    output_dir_metrics = Path("output/metrics")
    output_dir_metrics.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Risk', 'Medium Risk', 'High Risk'],
                yticklabels=['Low Risk', 'Medium Risk', 'High Risk'])
    plt.title('Gradient Boosting Confusion Matrix (Clinical DAIC-WOZ)')
    plt.ylabel('Actual Clinical State (True)')
    plt.xlabel('AI Predicted State (Pred)')
    plt.tight_layout()
    plt.savefig(output_dir_metrics / 'confusion_matrix.png', dpi=300)
    plt.close()

    # Feature Importance
    clf = pipeline.named_steps['classifier']
    importances = clf.feature_importances_
    print(f"\n   Feature Importance Ranking:")
    feat_imps = sorted(zip(available_features, importances), key=lambda x: -x[1])
    for feat, imp in feat_imps:
        bar = "█" * int(imp * 50)
        print(f"   {feat:>25}: {imp:.4f} {bar}")

    # ----- SAVE FEATURE IMPORTANCE PLOT -----
    plt.figure(figsize=(10, 8))
    feats = [x[0] for x in feat_imps][:15] # Top 15
    imps = [x[1] for x in feat_imps][:15]
    
    # Reverse so largest is at the top
    feats.reverse()
    imps.reverse()
    
    plt.barh(feats, imps, color='skyblue', edgecolor='black')
    plt.title('Top 15 Most Important Vocal Biomarkers (GBM)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(output_dir_metrics / 'feature_importance.png', dpi=300)
    plt.close()
    
    # ----- SAVE ROC CURVE (One-vs-Rest for 3 classes) -----
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = pipeline.predict_proba(X_test)
    
    class_names = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = ['#0984e3', '#00b894', '#d63031']
    
    plt.figure(figsize=(9, 7))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2.5,
                 label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve — Gradient Boosting (DAIC-WOZ Clinical)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir_metrics / 'roc_curve.png', dpi=300)
    plt.close()
    
    # ----- SAVE PRECISION-RECALL CURVE -----
    plt.figure(figsize=(9, 7))
    for i in range(3):
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall_vals, precision_vals, color=colors[i], lw=2.5,
                 label=f'{class_names[i]} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve — Gradient Boosting (DAIC-WOZ Clinical)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir_metrics / 'precision_recall_curve.png', dpi=300)
    plt.close()
    
    print("\n📈 Saved all metric diagrams to output/metrics/:")
    print("   • confusion_matrix.png")
    print("   • feature_importance.png")
    print("   • roc_curve.png")
    print("   • precision_recall_curve.png")

    # Save the model (this REPLACES the CREMA-D model)
    model_data = {
        'model': pipeline,
        'features': available_features,
        'classes': {0: 'Low', 1: 'Medium', 2: 'High'},
        'accuracy': acc,
        'dataset': 'DAIC-WOZ',
        'n_samples': len(df)
    }
    
    output_dir = Path("output/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as the PRIMARY model (replaces CREMA-D model)
    model_path = output_dir / "acoustic_risk_model.pkl"
    joblib.dump(model_data, model_path)
    
    print(f"\n{'='*60}")
    print(f"💾 CLINICAL MODEL SAVED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"   Path: {model_path}")
    print(f"   Model: GradientBoostingClassifier")
    print(f"   Dataset: DAIC-WOZ Clinical Interviews")
    print(f"   Samples: {len(df)}")
    print(f"   Accuracy: {acc*100:.2f}%")
    print(f"   Classes: Low (0) → Medium (1) → High (2)")
    print(f"\n   This model is now the PRIMARY model used by LangGraph.")
    print(f"   Run 'streamlit run source_codes/app_dashboard.py' to use it!")

if __name__ == "__main__":
    train_daic_woz_model()
