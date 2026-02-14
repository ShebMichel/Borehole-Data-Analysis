import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoreholeDataCleaner:
    """Clean and standardize borehole exploration data"""
    
    LITHOLOGY_MAP = {
        'clay': 'CLAY', 'silty clay': 'CLAY', 'sandy clay': 'CLAY',
        'sand': 'SAND', 'fine sand': 'SAND', 'medium sand': 'SAND', 'coarse sand': 'SAND',
        'gravel': 'GRAVEL', 'sandy gravel': 'GRAVEL',
        'silt': 'SILT',
        'rock': 'ROCK', 'bedrock': 'ROCK', 'weathered rock': 'ROCK'
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
        before = len(df)
        df = df.drop_duplicates(subset=['borehole_id', 'depth_m'])
        self.cleaning_stats['duplicates_removed'] = before - len(df)
        return df
    
    def _standardize_lithology(self, df: pd.DataFrame) -> pd.DataFrame:
        df['lithology'] = df['lithology'].str.lower().str.strip().map(self.LITHOLOGY_MAP)
        unmapped = df['lithology'].isnull().sum()
        if unmapped > 0:
            logger.warning(f"{unmapped} lithology values could not be mapped")
            df = df.dropna(subset=['lithology'])
        self.cleaning_stats['unmapped_lithology'] = unmapped
        return df
    
    def _fix_depth_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['depth_m'] >= 0]
        df = df.sort_values(['borehole_id', 'depth_m'])
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[(df['spt_n'] >= 0) & (df['spt_n'] <= 200)]
        if 'moisture' in df.columns:
            df = df[(df['moisture'] >= 0) & (df['moisture'] <= 100)]
        self.cleaning_stats['outliers_removed'] = before - len(df)
        return df
    
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if df['spt_n'].isnull().any():
            df['spt_n'] = df.groupby('lithology')['spt_n'].transform(lambda x: x.fillna(x.median()))
        if 'moisture' in df.columns and df['moisture'].isnull().any():
            df['moisture'] = df.groupby('lithology')['moisture'].transform(lambda x: x.fillna(x.mean()))
        return df
