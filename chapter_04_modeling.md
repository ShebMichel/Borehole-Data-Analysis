# Chapter 4: Building Predictive Models

## The Modeling Challenge

Predicting lithology from borehole data involves:
- **Imbalanced classes**: Rock samples are rare compared to clay/sand
- **Sequential dependencies**: Soil layers follow geological patterns
- **Spatial correlation**: Nearby boreholes have similar profiles
- **Limited samples**: Drilling is expensive, datasets are small

## Model Selection Strategy

```python
# src/models/model_selection.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from typing import Dict, List

CANDIDATE_MODELS = {
    'logistic': LogisticRegression(max_iter=1000, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'xgboost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
    'svm': SVC(kernel='rbf', probability=True, random_state=42)
}


def get_baseline_models() -> Dict:
    """Return dictionary of baseline models for comparison"""
    return CANDIDATE_MODELS.copy()
```

## Training Pipeline

```python
# src/models/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import json

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
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train/test split (stratified to maintain class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
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
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Classification metrics
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        self.metrics['test_accuracy'] = report['accuracy']
        self.metrics['test_f1_macro'] = report['macro avg']['f1-score']
        self.metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return y_pred, y_pred_proba
    
    def save(self, output_dir: Path):
        """Save model and artifacts"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, output_dir / 'model.pkl')
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, output_dir / 'label_encoder.pkl')
        
        # Save metadata
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


# Training script
if __name__ == '__main__':
    # Load prepared data
    df = pd.read_csv('data/processed/site_001_features.csv')
    
    # Define features
    feature_cols = [
        'depth_m', 'spt_n', 'moisture', 'relative_depth',
        'spt_rolling_mean', 'distance_from_origin'
    ]
    
    # Train XGBoost model
    from src.models.model_selection import CANDIDATE_MODELS
    
    classifier = LithologyClassifier(
        model=CANDIDATE_MODELS['xgboost'],
        feature_cols=feature_cols
    )
    
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    classifier.train(X_train, y_train, use_smote=True)
    y_pred, y_proba = classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save(Path('models/lithology_classifier_v1'))
```

## Hyperparameter Tuning

```python
# src/models/tune.py

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

class HyperparameterTuner:
    """Optimize model hyperparameters"""
    
    PARAM_GRIDS = {
        'xgboost': {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        },
        'random_forest': {
            'n_estimators': randint(50, 200),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    }
    
    def tune(self, model, model_name: str, X_train, y_train, n_iter: int = 50):
        """Run randomized search"""
        
        param_grid = self.PARAM_GRIDS.get(model_name, {})
        
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=5,
            scoring='f1_macro',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best F1 score: {search.best_score_:.3f}")
        
        return search.best_estimator_


# Example usage
if __name__ == '__main__':
    from xgboost import XGBClassifier
    
    df = pd.read_csv('data/processed/site_001_features.csv')
    
    feature_cols = ['depth_m', 'spt_n', 'moisture', 'relative_depth']
    classifier = LithologyClassifier(XGBClassifier(random_state=42), feature_cols)
    
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    X_train, y_train = classifier.handle_imbalance(X_train, y_train)
    
    tuner = HyperparameterTuner()
    best_model = tuner.tune(classifier.model, 'xgboost', X_train, y_train)
    
    classifier.model = best_model
    classifier.evaluate(X_test, y_test)
```

## Model Evaluation and Analysis

```python
# src/models/evaluate.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, classifier: LithologyClassifier):
        self.classifier = classifier
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path: str = None):
        """Plot confusion matrix heatmap"""
        
        cm = confusion_matrix(y_true, y_pred)
        classes = self.classifier.label_encoder.classes_
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes
        )
        plt.title('Confusion Matrix - Lithology Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, save_path: str = None):
        """Plot feature importance"""
        
        if not hasattr(self.classifier.model, 'feature_importances_'):
            print("Model does not support feature importance")
            return
        
        importances = self.classifier.model.feature_importances_
        features = self.classifier.feature_cols
        
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
        plt.title('Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test, save_path: str = None):
        """Plot ROC curves for multi-class classification"""
        
        y_score = self.classifier.model.predict_proba(X_test)
        classes = self.classifier.label_encoder.classes_
        n_classes = len(classes)
        
        # Binarize labels
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class Classification')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_errors(self, X_test, y_test, df_test: pd.DataFrame):
        """Analyze misclassification patterns"""
        
        y_pred = self.classifier.model.predict(X_test)
        
        # Find misclassified samples
        errors = y_test != y_pred
        error_indices = np.where(errors)[0]
        
        print(f"\n=== Error Analysis ===")
        print(f"Total errors: {errors.sum()} / {len(y_test)} ({100*errors.mean():.1f}%)")
        
        # Most confused pairs
        cm = confusion_matrix(y_test, y_pred)
        classes = self.classifier.label_encoder.classes_
        
        print("\nMost confused class pairs:")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i, j] > 0:
                    print(f"  {classes[i]} → {classes[j]}: {cm[i, j]} times")


# Example usage
if __name__ == '__main__':
    from pathlib import Path
    
    # Load trained model
    classifier = LithologyClassifier.load(Path('models/lithology_classifier_v1'))
    
    # Load test data
    df = pd.read_csv('data/processed/site_001_features.csv')
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Evaluate
    evaluator = ModelEvaluator(classifier)
    y_pred, _ = classifier.evaluate(X_test, y_test)
    
    evaluator.plot_confusion_matrix(y_test, y_pred, 'reports/figures/confusion_matrix.png')
    evaluator.plot_feature_importance('reports/figures/feature_importance.png')
    evaluator.plot_roc_curves(X_test, y_test, 'reports/figures/roc_curves.png')
```

## What Interviewers Look For

✅ **Baseline comparison**: Testing multiple algorithms before choosing one
✅ **Class imbalance handling**: Using SMOTE or class weights
✅ **Proper evaluation**: Cross-validation, multiple metrics, error analysis
✅ **Feature importance**: Understanding what drives predictions
✅ **Model versioning**: Saving models with metadata

❌ Only trying one algorithm
❌ Using accuracy alone for imbalanced data
❌ No analysis of failure modes
❌ Overfitting to training data

## Exercise

1. Train at least 3 different models on your data
2. Compare their F1-scores and training times
3. Tune hyperparameters for the best model
4. Create visualizations showing where your model fails
5. Document your model selection reasoning

---

**Next**: Chapter 5 - Deploying Geotechnical Systems
