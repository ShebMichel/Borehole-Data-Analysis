import numpy as np
import pandas as pd
from typing import List

class SyntheticBoreholeGenerator:
    """Generate realistic borehole data for testing"""
    
    SOIL_TYPES = ['CLAY', 'SAND', 'GRAVEL', 'SILT', 'ROCK']
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_borehole(self, borehole_id: str, max_depth: float = 30.0, interval: float = 1.5) -> pd.DataFrame:
        """Generate single borehole with depth-dependent properties"""
        
        depths = np.arange(0, max_depth, interval)
        lithology = []
        spt_values = []
        
        for depth in depths:
            if depth < 2:
                soil = 'CLAY'
                spt = np.random.randint(4, 10)
            elif depth < 10:
                soil = 'SAND' if np.random.random() > 0.3 else 'SILT'
                spt = np.random.randint(10, 30)
            elif depth < 20:
                soil = 'SAND' if np.random.random() > 0.4 else 'GRAVEL'
                spt = np.random.randint(20, 50)
            else:
                soil = 'ROCK' if np.random.random() > 0.2 else 'GRAVEL'
                spt = np.random.randint(50, 100) if soil == 'ROCK' else np.random.randint(30, 60)
            
            lithology.append(soil)
            spt_values.append(spt)
        
        n_samples = len(depths)
        return pd.DataFrame({
            'borehole_id': borehole_id,
            'depth_m': depths,
            'lithology': lithology,
            'spt_n': spt_values,
            'moisture': np.random.uniform(10, 30, n_samples),
            'color': np.random.choice(['BROWN', 'GRAY', 'RED', 'YELLOW'], n_samples)
        })
    
    def generate_site(self, n_boreholes: int = 10, site_name: str = 'SITE_001') -> pd.DataFrame:
        """Generate multiple boreholes for a site"""
        
        boreholes = []
        for i in range(n_boreholes):
            bh_id = f"{site_name}_BH{i+1:03d}"
            df = self.generate_borehole(bh_id)
            df['x_coord'] = 100 * (i % 5)
            df['y_coord'] = 100 * (i // 5)
            boreholes.append(df)
        
        return pd.concat(boreholes, ignore_index=True)
