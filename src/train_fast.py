"""
Fast Meme Classifier Training Script
=====================================
Trains models quickly without exhaustive hyperparameter search.
Uses optimized default parameters for fast training (~30 seconds).
"""

import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime
from time import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_PATH = "data/meme_features.csv"
MODEL_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(csv_path):
    """Load and prepare the dataset."""
    print(f"📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0)
    X = np.clip(X, -1e10, 1e10)
    
    print(f"   ✅ Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Class distribution:")
    for label, count in zip(unique, counts):
        print(f"   {label}: {count} samples")
    
    return X, y


def train_fast():
    """Train models quickly with optimized defaults."""
    start_time = time()
    
    # Load data
    X, y = load_data(CSV_PATH)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n🔀 Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # ========== 1. SVM (Fast - no grid search) ==========
    print("\n" + "="*50)
    print("🚀 Training SVM (optimized defaults)...")
    t0 = time()
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=50,           # Good default for gesture data
            gamma='scale',  # Auto-scale based on features
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_time = time() - t0
    
    # Evaluate SVM
    y_pred = svm_pipeline.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)
    svm_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"   ⏱️  Time: {svm_time:.2f}s")
    print(f"   📊 Accuracy: {svm_acc:.4f}")
    print(f"   📊 F1-Score: {svm_f1:.4f}")
    
    results['svm'] = {'accuracy': svm_acc, 'f1': svm_f1, 'time': svm_time, 'model': svm_pipeline}
    
    # ========== 2. Gradient Boosting (Skipped - too slow) ==========
    # Uncomment below if you want GB (adds ~4 minutes)
    # print("\n" + "="*50)
    # print("🚀 Training Gradient Boosting (fast config)...")
    # t0 = time()
    # gb_pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE))
    # ])
    # gb_pipeline.fit(X_train, y_train)
    # gb_time = time() - t0
    # y_pred = gb_pipeline.predict(X_test)
    # gb_acc = accuracy_score(y_test, y_pred)
    # gb_f1 = f1_score(y_test, y_pred, average='weighted')
    # print(f"   ⏱️  Time: {gb_time:.2f}s")
    # print(f"   📊 Accuracy: {gb_acc:.4f}")
    # print(f"   📊 F1-Score: {gb_f1:.4f}")
    # results['gb'] = {'accuracy': gb_acc, 'f1': gb_f1, 'time': gb_time, 'model': gb_pipeline}
    
    # ========== 3. Random Forest (Very Fast) ==========
    print("\n" + "="*50)
    print("🚀 Training Random Forest...")
    t0 = time()
    
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            n_jobs=-1,  # Use all CPU cores
            random_state=RANDOM_STATE
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_time = time() - t0
    
    # Evaluate RF
    y_pred = rf_pipeline.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    rf_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"   ⏱️  Time: {rf_time:.2f}s")
    print(f"   📊 Accuracy: {rf_acc:.4f}")
    print(f"   📊 F1-Score: {rf_f1:.4f}")
    
    results['rf'] = {'accuracy': rf_acc, 'f1': rf_f1, 'time': rf_time, 'model': rf_pipeline}
    
    # ========== Select Best Model ==========
    print("\n" + "="*50)
    print("🏆 MODEL COMPARISON:")
    print("="*50)
    
    best_name = max(results, key=lambda k: results[k]['f1'])
    
    for name, r in results.items():
        marker = "👑" if name == best_name else "  "
        print(f"{marker} {name.upper():4s} | Acc: {r['accuracy']:.4f} | F1: {r['f1']:.4f} | Time: {r['time']:.2f}s")
    
    best_model = results[best_name]['model']
    best_f1 = results[best_name]['f1']
    
    # ========== Save Best Model ==========
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save as main classifier
    model_path = os.path.join(MODEL_DIR, "meme_classifier.joblib")
    dump(best_model, model_path)
    print(f"\n💾 Saved best model ({best_name.upper()}) to: {model_path}")
    
    # Also save individual models
    dump(svm_pipeline, os.path.join(MODEL_DIR, "meme_svm_classifier.joblib"))
    
    # Save metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "best_model": best_name,
        "best_f1_score": float(best_f1),
        "best_accuracy": float(results[best_name]['accuracy']),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X.shape[1],
        "classes": list(np.unique(y)),
        "model_results": {
            name: {"accuracy": float(r['accuracy']), "f1": float(r['f1']), "time": float(r['time'])}
            for name, r in results.items()
        }
    }
    
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ========== Print Classification Report for Best Model ==========
    print(f"\n📋 Classification Report ({best_name.upper()}):")
    print("-"*50)
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))
    
    total_time = time() - start_time
    print(f"\n✅ Total training time: {total_time:.2f} seconds")
    print(f"🎉 Training complete!")
    
    return best_model


if __name__ == "__main__":
    train_fast()
