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
        return df
    
    def _add_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['relative_depth'] = df.groupby('borehole_id')['depth_m'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )
        return df
    
    def _add_spt_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['spt_rolling_mean'] = df.groupby('borehole_id')['spt_n'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        return df
    
    def _add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['distance_from_origin'] = np.sqrt(df['x_coord']**2 + df['y_coord']**2)
        return df
