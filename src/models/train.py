import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import json
from typing import List

class LithologyClassifier:
    """Train and evaluate lithology classification models"""
    
    def __init__(self, model, feature_cols: List[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.metrics = {}
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'lithology'):
        """Split and preprocess data"""
        X = df[self.feature_cols]
        y = df[target_col]
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE to balance classes"""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    def train(self, X_train, y_train, use_smote: bool = True):
        """Train the model"""
        if use_smote:
            X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        self.model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        self.metrics['test_accuracy'] = report['accuracy']
        self.metrics['test_f1_macro'] = report['macro avg']['f1-score']
        self.metrics['classification_report'] = report
        
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return y_pred, y_pred_proba
    
    def save(self, output_dir: Path):
        """Save model and artifacts"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, output_dir / 'model.pkl')
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, output_dir / 'label_encoder.pkl')
        
        metadata = {
            'feature_cols': self.feature_cols,
            'classes': self.label_encoder.classes_.tolist(),
            'metrics': self.metrics
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: Path):
        """Load trained model"""
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        model = joblib.load(model_dir / 'model.pkl')
        classifier = cls(model, metadata['feature_cols'])
        classifier.scaler = joblib.load(model_dir / 'scaler.pkl')
        classifier.label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
        classifier.metrics = metadata['metrics']
        
        return classifier
