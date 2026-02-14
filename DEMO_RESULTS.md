# Demo Run Results

## Execution Summary

**Date**: 2026-02-14  
**Status**: ✅ SUCCESS

## Pipeline Execution

The complete borehole analysis pipeline was successfully executed:

### 1. Data Generation
- Generated **400 samples** from **20 boreholes**
- Depth range: 0.0 - 28.5 meters
- 5 lithology types: SAND, ROCK, GRAVEL, CLAY, SILT

### 2. Data Cleaning
- No duplicates found
- All lithology values successfully mapped
- No outliers removed
- Dataset: 100% valid samples

### 3. Feature Engineering
- Created **11 features** including:
  - depth_m, spt_n, moisture
  - relative_depth (normalized within borehole)
  - spt_rolling_mean (3-sample window)
  - distance_from_origin (spatial feature)

### 4. Model Training
- **Algorithm**: Random Forest Classifier (100 trees)
- **Class balancing**: SMOTE applied
- **Cross-validation**: 5-fold CV
- **Training samples**: 320 (after 80/20 split)

### 5. Model Performance

#### Overall Metrics
- **Accuracy**: 71.3%
- **F1-Score (macro)**: 73.8%
- **CV Score**: 81.1% (±3.7%)

#### Per-Class Performance
| Lithology | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| CLAY      | 1.00      | 1.00   | 1.00     | 8       |
| GRAVEL    | 0.50      | 0.69   | 0.58     | 16      |
| ROCK      | 0.89      | 0.89   | 0.89     | 19      |
| SAND      | 0.76      | 0.53   | 0.63     | 30      |
| SILT      | 0.50      | 0.71   | 0.59     | 7       |

## Generated Files

```
src/
├── data/
│   ├── raw/
│   │   └── demo_site.csv                    (400 samples)
│   └── processed/
│       ├── demo_site_clean.csv              (cleaned data)
│       └── demo_site_features.csv           (with engineered features)
└── models/
    └── demo_classifier/
        ├── model.pkl                        (trained Random Forest)
        ├── scaler.pkl                       (StandardScaler)
        ├── label_encoder.pkl                (label mappings)
        └── metadata.json                    (model info & metrics)
```

## Key Insights

### Strengths
- **CLAY classification**: Perfect precision and recall (100%)
- **ROCK classification**: Strong performance (89% F1-score)
- **Model stability**: Low CV standard deviation (3.7%)

### Areas for Improvement
- **GRAVEL vs SAND confusion**: Model struggles to distinguish these classes
- **SILT classification**: Low precision (50%) - often confused with SAND
- **Class imbalance**: Original dataset had uneven distribution (SAND: 152, SILT: 35)

### Recommendations
1. Collect more SILT and CLAY samples to improve balance
2. Add grain size distribution features to better separate SAND/GRAVEL/SILT
3. Consider ensemble methods or deep learning for better performance
4. Implement confidence thresholds for production deployment

## Technical Notes

- **Runtime**: ~5 seconds on standard laptop
- **Memory usage**: < 100MB
- **Dependencies**: pandas, numpy, scikit-learn, imbalanced-learn
- **Python version**: 3.13

## Next Steps

1. ✅ Data generation and model training complete
2. 🔄 Deploy API endpoint (see chapter_05_deployment.md)
3. 🔄 Create interactive visualizations (see chapter_06_communication.md)
4. 🔄 Set up monitoring and logging
5. 🔄 Deploy to AWS/cloud platform

---

**Conclusion**: The borehole lithology classification system is functional and ready for further development. The model achieves reasonable performance (74% F1-score) and can be improved with additional features and more balanced training data.
