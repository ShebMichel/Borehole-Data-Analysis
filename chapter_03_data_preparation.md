# Chapter 3: Data Validation and Standardization

## Why Data Preparation Matters More in Geotechnical Work

Borehole data comes from field conditions where:
- Drillers use different terminology ("sandy clay" vs "clayey sand")
- Equipment malfunctions produce impossible readings
- Depths are recorded inconsistently (meters vs feet, ground level vs datum)
- Missing values mean different things (not tested vs test failed)

## Exploratory Data Analysis

```python
# notebooks/01_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_profile(filepath: str) -> pd.DataFrame:
    """Load data and generate quality report"""
    
    df = pd.read_csv(filepath)
    
    print("=== Dataset Profile ===")
    print(f"Total records: {len(df)}")
    print(f"Unique boreholes: {df['borehole_id'].nunique()}")
    print(f"Depth range: {df['depth_m'].min():.1f} - {df['depth_m'].max():.1f} m")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nLithology distribution:\n{df['lithology'].value_counts()}")
    
    return df


def plot_data_quality(df: pd.DataFrame):
    """Visualize data quality issues"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # SPT distribution by lithology
    df.boxplot(column='spt_n', by='lithology', ax=axes[0, 0])
    axes[0, 0].set_title('SPT Values by Soil Type')
    axes[0, 0].set_ylabel('SPT N-value')
    
    # Depth distribution
    axes[0, 1].hist(df['depth_m'], bins=30, edgecolor='black')
    axes[0, 1].set_title('Sampling Depth Distribution')
    axes[0, 1].set_xlabel('Depth (m)')
    
    # Missing data heatmap
    sns.heatmap(df.isnull(), cbar=False, ax=axes[1, 0])
    axes[1, 0].set_title('Missing Data Pattern')
    
    # Correlation matrix
    numeric_cols = ['depth_m', 'spt_n', 'moisture']
    sns.heatmap(df[numeric_cols].corr(), annot=True, ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlations')
    
    plt.tight_layout()
    plt.savefig('reports/figures/data_quality.png', dpi=300)


if __name__ == '__main__':
    df = load_and_profile('data/raw/synthetic_site_001.csv')
    plot_data_quality(df)
```

## Data Cleaning Pipeline

```python
# src/preprocessing/cleaner.py

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoreholeDataCleaner:
    """Clean and standardize borehole exploration data"""
    
    # Standardized lithology mapping
    LITHOLOGY_MAP = {
        'clay': 'CLAY',
        'silty clay': 'CLAY',
        'sandy clay': 'CLAY',
        'sand': 'SAND',
        'fine sand': 'SAND',
        'medium sand': 'SAND',
        'coarse sand': 'SAND',
        'gravel': 'GRAVEL',
        'sandy gravel': 'GRAVEL',
        'silt': 'SILT',
        'rock': 'ROCK',
        'bedrock': 'ROCK',
        'weathered rock': 'ROCK'
    }
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps"""
        
        df = df.copy()
        
        df = self._remove_duplicates(df)
        df = self._standardize_lithology(df)
        df = self._fix_depth_issues(df)
        df = self._handle_outliers(df)
        df = self._impute_missing(df)
        
        logger.info(f"Cleaning complete: {self.cleaning_stats}")
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate depth measurements in same borehole"""
        
        before = len(df)
        df = df.drop_duplicates(subset=['borehole_id', 'depth_m'])
        after = len(df)
        
        self.cleaning_stats['duplicates_removed'] = before - after
        return df
    
    def _standardize_lithology(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map various lithology terms to standard categories"""
        
        df['lithology'] = (
            df['lithology']
            .str.lower()
            .str.strip()
            .map(self.LITHOLOGY_MAP)
        )
        
        # Handle unmapped values
        unmapped = df['lithology'].isnull().sum()
        if unmapped > 0:
            logger.warning(f"{unmapped} lithology values could not be mapped")
            df = df.dropna(subset=['lithology'])
        
        self.cleaning_stats['unmapped_lithology'] = unmapped
        return df
    
    def _fix_depth_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix common depth recording errors"""
        
        # Remove negative depths
        df = df[df['depth_m'] >= 0]
        
        # Sort by borehole and depth
        df = df.sort_values(['borehole_id', 'depth_m'])
        
        # Flag unrealistic depth jumps (>10m between samples)
        df['depth_jump'] = df.groupby('borehole_id')['depth_m'].diff()
        suspicious = (df['depth_jump'] > 10).sum()
        
        if suspicious > 0:
            logger.warning(f"{suspicious} samples with >10m depth jumps")
        
        df = df.drop(columns=['depth_jump'])
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove physically impossible values"""
        
        # SPT values should be 0-200
        before = len(df)
        df = df[(df['spt_n'] >= 0) & (df['spt_n'] <= 200)]
        
        # Moisture content 0-100%
        if 'moisture' in df.columns:
            df = df[(df['moisture'] >= 0) & (df['moisture'] <= 100)]
        
        self.cleaning_stats['outliers_removed'] = before - len(df)
        return df
    
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values strategically"""
        
        # SPT: use median by lithology
        if df['spt_n'].isnull().any():
            df['spt_n'] = df.groupby('lithology')['spt_n'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Moisture: use mean by lithology
        if 'moisture' in df.columns and df['moisture'].isnull().any():
            df['moisture'] = df.groupby('lithology')['moisture'].transform(
                lambda x: x.fillna(x.mean())
            )
        
        return df


# Example usage
if __name__ == '__main__':
    df = pd.read_csv('data/raw/synthetic_site_001.csv')
    
    cleaner = BoreholeDataCleaner()
    df_clean = cleaner.clean(df)
    
    df_clean.to_csv('data/processed/site_001_clean.csv', index=False)
    print(f"Cleaned data: {len(df_clean)} records")
```

## Feature Engineering

```python
# src/preprocessing/features.py

import pandas as pd
import numpy as np

class BoreholeFeatureEngineer:
    """Create features for ML models"""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features"""
        
        df = df.copy()
        
        df = self._add_depth_features(df)
        df = self._add_spt_features(df)
        df = self._add_spatial_features(df)
        df = self._encode_categorical(df)
        
        return df
    
    def _add_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Depth-based features"""
        
        # Depth bins (shallow, medium, deep)
        df['depth_category'] = pd.cut(
            df['depth_m'],
            bins=[0, 5, 15, 100],
            labels=['shallow', 'medium', 'deep']
        )
        
        # Relative depth within borehole
        df['relative_depth'] = df.groupby('borehole_id')['depth_m'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )
        
        return df
    
    def _add_spt_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """SPT-derived features"""
        
        # Soil consistency from SPT (standard geotechnical interpretation)
        def spt_to_consistency(spt):
            if spt < 4:
                return 'very_soft'
            elif spt < 10:
                return 'soft'
            elif spt < 30:
                return 'medium'
            elif spt < 50:
                return 'stiff'
            else:
                return 'very_stiff'
        
        df['consistency'] = df['spt_n'].apply(spt_to_consistency)
        
        # Rolling average SPT (3-sample window)
        df['spt_rolling_mean'] = df.groupby('borehole_id')['spt_n'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        return df
    
    def _add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spatial context features"""
        
        # Distance from site origin
        df['distance_from_origin'] = np.sqrt(
            df['x_coord']**2 + df['y_coord']**2
        )
        
        # Borehole density (number of nearby boreholes)
        # Simplified: count boreholes within 50m
        def count_nearby(row, df_all):
            dist = np.sqrt(
                (df_all['x_coord'] - row['x_coord'])**2 +
                (df_all['y_coord'] - row['y_coord'])**2
            )
            return (dist < 50).sum() - 1  # Exclude self
        
        unique_bh = df.drop_duplicates('borehole_id')
        df['nearby_boreholes'] = df.apply(
            lambda row: count_nearby(row, unique_bh), axis=1
        )
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        
        # One-hot encode color
        if 'color' in df.columns:
            color_dummies = pd.get_dummies(df['color'], prefix='color')
            df = pd.concat([df, color_dummies], axis=1)
        
        return df


# Example usage
if __name__ == '__main__':
    df = pd.read_csv('data/processed/site_001_clean.csv')
    
    engineer = BoreholeFeatureEngineer()
    df_features = engineer.transform(df)
    
    df_features.to_csv('data/processed/site_001_features.csv', index=False)
    print(f"Features created: {df_features.columns.tolist()}")
```

## Data Validation Tests

```python
# tests/test_preprocessing.py

import pytest
import pandas as pd
from src.preprocessing.cleaner import BoreholeDataCleaner

def test_lithology_standardization():
    """Test lithology mapping works correctly"""
    
    df = pd.DataFrame({
        'borehole_id': ['BH1', 'BH1', 'BH1'],
        'depth_m': [1.5, 3.0, 4.5],
        'lithology': ['Sandy Clay', 'SAND', 'gravel'],
        'spt_n': [8, 15, 25],
        'x_coord': [0, 0, 0],
        'y_coord': [0, 0, 0]
    })
    
    cleaner = BoreholeDataCleaner()
    df_clean = cleaner.clean(df)
    
    assert df_clean['lithology'].tolist() == ['CLAY', 'SAND', 'GRAVEL']


def test_outlier_removal():
    """Test impossible SPT values are removed"""
    
    df = pd.DataFrame({
        'borehole_id': ['BH1', 'BH1', 'BH1'],
        'depth_m': [1.5, 3.0, 4.5],
        'lithology': ['CLAY', 'SAND', 'GRAVEL'],
        'spt_n': [8, 250, 25],  # 250 is impossible
        'x_coord': [0, 0, 0],
        'y_coord': [0, 0, 0]
    })
    
    cleaner = BoreholeDataCleaner()
    df_clean = cleaner.clean(df)
    
    assert len(df_clean) == 2
    assert 250 not in df_clean['spt_n'].values


def test_duplicate_removal():
    """Test duplicate depth measurements are removed"""
    
    df = pd.DataFrame({
        'borehole_id': ['BH1', 'BH1', 'BH1'],
        'depth_m': [1.5, 1.5, 3.0],  # Duplicate at 1.5m
        'lithology': ['CLAY', 'CLAY', 'SAND'],
        'spt_n': [8, 8, 15],
        'x_coord': [0, 0, 0],
        'y_coord': [0, 0, 0]
    })
    
    cleaner = BoreholeDataCleaner()
    df_clean = cleaner.clean(df)
    
    assert len(df_clean) == 2
```

## What Interviewers Look For

✅ **Domain knowledge**: Understanding what values are physically possible
✅ **Systematic approach**: Cleaning pipeline, not ad-hoc fixes
✅ **Testing**: Unit tests for data transformations
✅ **Documentation**: Explaining why certain cleaning decisions were made

❌ Dropping all rows with missing values
❌ No validation after cleaning
❌ Hardcoded values without explanation

## Exercise

1. Run the EDA script on your synthetic data
2. Identify at least 3 data quality issues
3. Implement custom cleaning logic for your specific issues
4. Write tests to verify your cleaning works correctly

---

**Next**: Chapter 4 - Building Predictive Models
