#!/usr/bin/env python3
"""
Quick start demo for borehole lithology classification system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic_generator import SyntheticBoreholeGenerator
from preprocessing.cleaner import BoreholeDataCleaner
from preprocessing.features import BoreholeFeatureEngineer
from models.train import LithologyClassifier
from models.model_selection import CANDIDATE_MODELS
import pandas as pd


def run_demo():
    """Run complete pipeline demonstration"""
    
    print("=" * 80)
    print("BOREHOLE LITHOLOGY CLASSIFICATION - DEMO")
    print("=" * 80)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic borehole data...")
    generator = SyntheticBoreholeGenerator(seed=42)
    df_raw = generator.generate_site(n_boreholes=20, site_name='DEMO_SITE')
    print(f"✓ Generated {len(df_raw)} samples from 20 boreholes")
    
    # Save raw data
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    df_raw.to_csv('data/raw/demo_site.csv', index=False)
    
    # Step 2: Clean data
    print("\n[2/5] Cleaning and validating data...")
    cleaner = BoreholeDataCleaner()
    df_clean = cleaner.clean(df_raw)
    print(f"✓ Cleaned data: {len(df_clean)} valid samples")
    print(f"  Cleaning stats: {cleaner.cleaning_stats}")
    
    # Save cleaned data
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df_clean.to_csv('data/processed/demo_site_clean.csv', index=False)
    
    # Step 3: Engineer features
    print("\n[3/5] Engineering features...")
    engineer = BoreholeFeatureEngineer()
    df_features = engineer.transform(df_clean)
    print(f"✓ Created {len(df_features.columns)} features")
    
    # Save feature data
    df_features.to_csv('data/processed/demo_site_features.csv', index=False)
    
    # Step 4: Train model
    print("\n[4/5] Training Random Forest classifier...")
    feature_cols = [
        'depth_m', 'spt_n', 'moisture', 'relative_depth',
        'spt_rolling_mean', 'distance_from_origin'
    ]
    
    classifier = LithologyClassifier(
        model=CANDIDATE_MODELS['random_forest'],
        feature_cols=feature_cols
    )
    
    X_train, X_test, y_train, y_test = classifier.prepare_data(df_features)
    classifier.train(X_train, y_train, use_smote=True)
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating model...")
    y_pred, y_proba = classifier.evaluate(X_test, y_test)
    
    # Save model
    Path('models').mkdir(parents=True, exist_ok=True)
    classifier.save(Path('models/demo_classifier'))
    print(f"\n✓ Model saved to models/demo_classifier/")
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print(f"\nModel Performance:")
    print(f"  - Accuracy: {classifier.metrics['test_accuracy']:.3f}")
    print(f"  - F1-Score (macro): {classifier.metrics['test_f1_macro']:.3f}")
    print(f"  - CV Score: {classifier.metrics['cv_mean']:.3f} (+/- {classifier.metrics['cv_std']:.3f})")
    
    print(f"\nGenerated Files:")
    print(f"  - data/raw/demo_site.csv")
    print(f"  - data/processed/demo_site_clean.csv")
    print(f"  - data/processed/demo_site_features.csv")
    print(f"  - models/demo_classifier/")
    
    print(f"\nNext Steps:")
    print(f"  1. Explore notebooks/ for detailed analysis")
    print(f"  2. Run API: python src/api/main.py")
    print(f"  3. View documentation in docs/")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Borehole Analysis Demo')
    parser.add_argument('--demo', action='store_true', help='Run full demo pipeline')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        print("Usage: python src/main.py --demo")
        print("Run the complete borehole analysis pipeline demonstration")
