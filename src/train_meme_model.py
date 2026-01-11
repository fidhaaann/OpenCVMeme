"""
Meme Classifier Training Script
================================
Train a high-confidence classifier to distinguish 6 memes based on
facial expressions and dual-hand gestures.

Supports:
- SVM with RBF kernel (default)
- Gradient Boosting Classifier
- Random Forest (baseline)

Features:
- Automatic hyperparameter tuning
- Cross-validation
- Probability calibration
- Model evaluation and confusion matrix
"""

import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# INSERT DATASET DIRECTORY HERE
CSV_PATH = "data/meme_features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "meme_classifier.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Minimum samples required per class
MIN_SAMPLES_PER_CLASS = 30


class MemeClassifierTrainer:
    """
    Trains and evaluates meme gesture classifiers.
    """
    
    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = csv_path
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.scaler = StandardScaler()
        
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def load_data(self):
        """Load and validate the dataset."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        
        print(f"📂 Loading dataset from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        if 'label' not in df.columns:
            raise ValueError("Dataset must have a 'label' column")
        
        self.X = df.drop('label', axis=1).values
        self.y = df['label'].values
        
        # Dataset statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {len(self.X)}")
        print(f"   Feature dimension: {self.X.shape[1]}")
        print(f"\n   Class distribution:")
        
        unique, counts = np.unique(self.y, return_counts=True)
        for label, count in zip(unique, counts):
            status = "✅" if count >= MIN_SAMPLES_PER_CLASS else "⚠️"
            print(f"   {status} {label}: {count} samples")
        
        # Check for NaN values
        nan_count = np.isnan(self.X).sum()
        if nan_count > 0:
            print(f"\n⚠️ Found {nan_count} NaN values, replacing with 0")
            self.X = np.nan_to_num(self.X, nan=0.0)
        
        # Check for infinite values
        inf_count = np.isinf(self.X).sum()
        if inf_count > 0:
            print(f"⚠️ Found {inf_count} infinite values, clipping")
            self.X = np.clip(self.X, -1e10, 1e10)
        
        return self
    
    def prepare_data(self, test_size=TEST_SIZE):
        """Split data into train and test sets."""
        print(f"\n🔀 Splitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=self.y
        )
        
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        
        return self
    
    def train_svm(self, tune_hyperparams=True):
        """
        Train SVM classifier with RBF kernel.
        
        Args:
            tune_hyperparams: Whether to perform grid search
            
        Returns:
            Trained pipeline
        """
        print("\n🎯 Training SVM Classifier...")
        
        if tune_hyperparams:
            print("   Performing hyperparameter tuning...")
            
            # Pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
            ])
            
            # Parameter grid - optimized for gesture sensitivity
            param_grid = {
                'svm__C': [1, 10, 50, 100],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
            }
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\n   Best parameters: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        
        else:
            # Default parameters
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    kernel='rbf',
                    C=50,
                    gamma='scale',
                    probability=True,
                    random_state=RANDOM_STATE
                ))
            ])
            
            pipeline.fit(self.X_train, self.y_train)
            return pipeline
    
    def train_gradient_boosting(self, tune_hyperparams=True):
        """
        Train Gradient Boosting classifier.
        
        Args:
            tune_hyperparams: Whether to perform grid search
            
        Returns:
            Trained pipeline
        """
        print("\n🎯 Training Gradient Boosting Classifier...")
        
        if tune_hyperparams:
            print("   Performing hyperparameter tuning...")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('gb', GradientBoostingClassifier(random_state=RANDOM_STATE))
            ])
            
            param_grid = {
                'gb__n_estimators': [100, 200, 300],
                'gb__learning_rate': [0.05, 0.1, 0.2],
                'gb__max_depth': [3, 5, 7],
                'gb__min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\n   Best parameters: {grid_search.best_params_}")
            print(f"   Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=5,
                    random_state=RANDOM_STATE
                ))
            ])
            
            pipeline.fit(self.X_train, self.y_train)
            return pipeline
    
    def train_random_forest(self):
        """
        Train Random Forest classifier (baseline).
        
        Returns:
            Trained pipeline
        """
        print("\n🎯 Training Random Forest Classifier (baseline)...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        return pipeline
    
    def evaluate_model(self, model, model_name="Model"):
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model/pipeline
            model_name: Name for display
            
        Returns:
            dict with evaluation metrics
        """
        print(f"\n📈 Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, self.X, self.y, 
            cv=CV_FOLDS, 
            scoring='f1_weighted'
        )
        
        print(f"\n   Test Accuracy: {accuracy:.4f}")
        print(f"   Test F1 Score: {f1:.4f}")
        print(f"   CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Classification report
        print(f"\n   Classification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        
        # Confidence analysis
        max_probs = np.max(y_pred_proba, axis=1)
        print(f"\n   Confidence Statistics:")
        print(f"   Mean confidence: {max_probs.mean():.4f}")
        print(f"   Min confidence: {max_probs.min():.4f}")
        print(f"   Max confidence: {max_probs.max():.4f}")
        print(f"   Predictions >= 0.85: {(max_probs >= 0.85).sum()} / {len(max_probs)}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        classes = model.classes_
        print(f"   {classes}")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'mean_confidence': max_probs.mean(),
            'confusion_matrix': cm.tolist()
        }
    
    def train_all_and_select_best(self, tune_hyperparams=True):
        """
        Train all classifier types and select the best one.
        
        Args:
            tune_hyperparams: Whether to tune hyperparameters
            
        Returns:
            Best model
        """
        print("\n" + "="*60)
        print("🏆 TRAINING ALL CLASSIFIERS AND SELECTING BEST")
        print("="*60)
        
        models = {}
        results = {}
        
        # Train SVM
        models['SVM'] = self.train_svm(tune_hyperparams)
        results['SVM'] = self.evaluate_model(models['SVM'], 'SVM')
        
        # Train Gradient Boosting
        models['GradientBoosting'] = self.train_gradient_boosting(tune_hyperparams)
        results['GradientBoosting'] = self.evaluate_model(models['GradientBoosting'], 'Gradient Boosting')
        
        # Train Random Forest (baseline)
        models['RandomForest'] = self.train_random_forest()
        results['RandomForest'] = self.evaluate_model(models['RandomForest'], 'Random Forest')
        
        # Select best model based on F1 score
        best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        self.best_model = models[best_name]
        
        print("\n" + "="*60)
        print(f"🥇 BEST MODEL: {best_name}")
        print(f"   F1 Score: {results[best_name]['f1_score']:.4f}")
        print(f"   Accuracy: {results[best_name]['accuracy']:.4f}")
        print("="*60)
        
        return self.best_model, best_name, results
    
    def calibrate_model(self, model):
        """
        Calibrate model probabilities for more reliable confidence scores.
        
        Args:
            model: Trained model
            
        Returns:
            Calibrated model
        """
        print("\n🔧 Calibrating model probabilities...")
        
        calibrated = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv=3
        )
        calibrated.fit(self.X_train, self.y_train)
        
        return calibrated
    
    def save_model(self, model, model_name='best', metadata=None):
        """
        Save the trained model and metadata.
        
        Args:
            model: Trained model to save
            model_name: Name identifier for the model
            metadata: Additional metadata to save
        """
        # Save model
        model_path = os.path.join(MODEL_DIR, f"meme_{model_name}_classifier.joblib")
        dump(model, model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Also save as default
        dump(model, MODEL_PATH)
        print(f"   Default model: {MODEL_PATH}")
        
        # Save metadata
        meta = {
            'model_name': model_name,
            'trained_at': datetime.now().isoformat(),
            'classes': list(model.classes_),
            'feature_dim': self.X.shape[1],
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
        }
        
        if metadata:
            meta.update(metadata)
        
        with open(METADATA_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"   Metadata: {METADATA_PATH}")
    
    def run_full_pipeline(self, classifier_type='auto', tune_hyperparams=True, calibrate=True):
        """
        Run the complete training pipeline.
        
        Args:
            classifier_type: 'svm', 'gb', 'rf', or 'auto' for best
            tune_hyperparams: Whether to tune hyperparameters
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Trained model
        """
        print("\n" + "="*60)
        print("🚀 MEME CLASSIFIER TRAINING PIPELINE")
        print("="*60)
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train model(s)
        if classifier_type == 'auto':
            model, model_name, results = self.train_all_and_select_best(tune_hyperparams)
            metadata = {
                'best_model_type': model_name,
                'all_results': {k: {'accuracy': v['accuracy'], 'f1': v['f1_score']} 
                               for k, v in results.items()}
            }
        elif classifier_type == 'svm':
            model = self.train_svm(tune_hyperparams)
            model_name = 'SVM'
            results = self.evaluate_model(model, model_name)
            metadata = {'accuracy': results['accuracy'], 'f1': results['f1_score']}
        elif classifier_type == 'gb':
            model = self.train_gradient_boosting(tune_hyperparams)
            model_name = 'GradientBoosting'
            results = self.evaluate_model(model, model_name)
            metadata = {'accuracy': results['accuracy'], 'f1': results['f1_score']}
        elif classifier_type == 'rf':
            model = self.train_random_forest()
            model_name = 'RandomForest'
            results = self.evaluate_model(model, model_name)
            metadata = {'accuracy': results['accuracy'], 'f1': results['f1_score']}
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Calibrate if requested
        if calibrate and classifier_type != 'auto':
            model = self.calibrate_model(model)
            print("   Model calibrated for better probability estimates")
        
        # Save model
        self.save_model(model, model_name.lower(), metadata)
        
        print("\n✅ Training complete!")
        
        return model


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Meme Gesture Classifier")
    parser.add_argument("--data", type=str, default=CSV_PATH,
                        help="Path to training data CSV")
    parser.add_argument("--model", type=str, choices=['auto', 'svm', 'gb', 'rf'],
                        default='auto',
                        help="Classifier type (auto=try all and pick best)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip probability calibration")
    
    args = parser.parse_args()
    
    trainer = MemeClassifierTrainer(csv_path=args.data)
    trainer.run_full_pipeline(
        classifier_type=args.model,
        tune_hyperparams=not args.no_tune,
        calibrate=not args.no_calibrate
    )


if __name__ == "__main__":
    main()
