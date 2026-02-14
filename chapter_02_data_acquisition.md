# Chapter 2: Acquiring Borehole Data

## The Data Challenge in Geotechnical Engineering

Unlike image or text datasets, borehole data is:
- **Fragmented**: Scattered across PDFs, Excel files, and proprietary formats
- **Inconsistent**: No universal standard for logging practices
- **Proprietary**: Companies guard their subsurface data
- **Sparse**: Boreholes are expensive ($5k-$50k each)

## Data Sources for Portfolio Projects

### Public Geological Surveys

Many government agencies provide free borehole data:

**United States**
- USGS National Water Information System
- State geological surveys (California, Texas, etc.)

**United Kingdom**
- British Geological Survey (BGS) OpenGeoscience

**Australia**
- Geoscience Australia
- State databases (NSW, Victoria, Queensland)

**Canada**
- Natural Resources Canada GeoGratis

### Academic Repositories

```python
# src/data/sources.py

OPEN_DATA_SOURCES = {
    'bgs_uk': {
        'url': 'https://www.bgs.ac.uk/datasets/boreholes/',
        'format': 'CSV',
        'fields': ['depth', 'lithology', 'location', 'spt_n'],
        'license': 'Open Government License'
    },
    'usgs': {
        'url': 'https://waterdata.usgs.gov/nwis',
        'format': 'JSON/CSV',
        'fields': ['depth', 'description', 'coordinates'],
        'license': 'Public Domain'
    }
}
```

### Creating Synthetic Data

For demonstration purposes, generate realistic synthetic boreholes:

```python
# src/data/synthetic_generator.py

import numpy as np
import pandas as pd
from typing import List, Tuple

class SyntheticBoreholeGenerator:
    """Generate realistic borehole data for testing"""
    
    SOIL_TYPES = ['CLAY', 'SAND', 'GRAVEL', 'SILT', 'ROCK']
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_borehole(self, 
                         borehole_id: str,
                         max_depth: float = 30.0,
                         interval: float = 1.5) -> pd.DataFrame:
        """Generate single borehole with depth-dependent properties"""
        
        depths = np.arange(0, max_depth, interval)
        n_samples = len(depths)
        
        # Simulate typical soil profile: topsoil -> clay -> sand -> rock
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
        
        return pd.DataFrame({
            'borehole_id': borehole_id,
            'depth_m': depths,
            'lithology': lithology,
            'spt_n': spt_values,
            'moisture': np.random.uniform(10, 30, n_samples),
            'color': np.random.choice(['BROWN', 'GRAY', 'RED', 'YELLOW'], n_samples)
        })
    
    def generate_site(self, 
                     n_boreholes: int = 10,
                     site_name: str = 'SITE_001') -> pd.DataFrame:
        """Generate multiple boreholes for a site"""
        
        boreholes = []
        for i in range(n_boreholes):
            bh_id = f"{site_name}_BH{i+1:03d}"
            df = self.generate_borehole(bh_id)
            
            # Add spatial coordinates (simulate grid pattern)
            df['x_coord'] = 100 * (i % 5)
            df['y_coord'] = 100 * (i // 5)
            
            boreholes.append(df)
        
        return pd.concat(boreholes, ignore_index=True)


# Example usage
if __name__ == '__main__':
    generator = SyntheticBoreholeGenerator()
    site_data = generator.generate_site(n_boreholes=15)
    site_data.to_csv('data/raw/synthetic_site_001.csv', index=False)
    print(f"Generated {len(site_data)} samples across 15 boreholes")
```

## Real Dataset: British Geological Survey

Let's work with actual BGS data:

```python
# src/data/bgs_loader.py

import pandas as pd
import requests
from pathlib import Path

class BGSDataLoader:
    """Load and parse British Geological Survey borehole data"""
    
    BASE_URL = "https://www.bgs.ac.uk/opengeoscience/api"
    
    def download_boreholes(self, 
                          bbox: Tuple[float, float, float, float],
                          output_path: Path) -> pd.DataFrame:
        """
        Download boreholes within bounding box
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            output_path: Where to save CSV
        """
        
        # Note: This is a simplified example
        # Real BGS API requires authentication and has rate limits
        
        params = {
            'bbox': ','.join(map(str, bbox)),
            'format': 'json'
        }
        
        # In production, add error handling and rate limiting
        response = requests.get(f"{self.BASE_URL}/boreholes", params=params)
        data = response.json()
        
        df = pd.DataFrame(data['features'])
        df.to_csv(output_path, index=False)
        
        return df
    
    def parse_lithology_log(self, log_text: str) -> List[dict]:
        """Parse free-text lithology descriptions"""
        
        # Simple parser - in production, use NLP
        entries = []
        for line in log_text.split('\n'):
            if '-' in line:
                depth_range, description = line.split('-', 1)
                entries.append({
                    'depth_from': float(depth_range.split()[0]),
                    'depth_to': float(depth_range.split()[1]),
                    'description': description.strip()
                })
        
        return entries
```

## Data Collection Best Practices

### 1. Document Data Provenance

```python
# data/metadata.json

{
  "dataset_name": "uk_bgs_london_basin",
  "source": "British Geological Survey",
  "url": "https://www.bgs.ac.uk/datasets/boreholes/",
  "license": "Open Government License v3.0",
  "date_accessed": "2026-02-14",
  "geographic_extent": {
    "region": "London Basin",
    "bbox": [-0.5, 51.3, 0.3, 51.7]
  },
  "n_boreholes": 247,
  "depth_range_m": [5, 150],
  "primary_use": "Foundation design and tunneling"
}
```

### 2. Version Your Data

```bash
# Use DVC (Data Version Control)
pip install dvc

dvc init
dvc add data/raw/bgs_london_basin.csv
git add data/raw/bgs_london_basin.csv.dvc .dvc/config
git commit -m "Add BGS London Basin dataset v1.0"
```

### 3. Validate on Ingestion

```python
# src/data/validation.py

from pydantic import BaseModel, Field, validator
from typing import Literal

class BoreholeRecord(BaseModel):
    """Schema for borehole data validation"""
    
    borehole_id: str = Field(..., min_length=3)
    depth_m: float = Field(..., ge=0, le=500)
    lithology: Literal['CLAY', 'SAND', 'GRAVEL', 'SILT', 'ROCK']
    spt_n: int = Field(..., ge=0, le=200)
    x_coord: float
    y_coord: float
    
    @validator('spt_n')
    def spt_realistic(cls, v, values):
        """Check SPT values match lithology expectations"""
        lithology = values.get('lithology')
        
        if lithology == 'CLAY' and v > 30:
            raise ValueError(f"SPT {v} too high for clay")
        if lithology == 'ROCK' and v < 50:
            raise ValueError(f"SPT {v} too low for rock")
        
        return v


def validate_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate entire dataset, return clean data and errors"""
    
    valid_records = []
    errors = []
    
    for idx, row in df.iterrows():
        try:
            record = BoreholeRecord(**row.to_dict())
            valid_records.append(row)
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
    
    return pd.DataFrame(valid_records), errors
```

## What Interviewers Look For

✅ **Data provenance**: You can explain where data came from and its limitations
✅ **Licensing awareness**: You checked if data can be used publicly
✅ **Validation logic**: You don't blindly trust input data
✅ **Reproducibility**: Others can re-download/regenerate your dataset

❌ Using data without attribution
❌ No validation or quality checks
❌ Mixing incompatible data sources without normalization

## Exercise

1. Generate a synthetic dataset with 20 boreholes using the provided code
2. Create a `metadata.json` file documenting your data
3. Write a validation function that checks for:
   - Depth values in ascending order
   - SPT values within realistic ranges
   - No missing lithology labels

---

**Next**: Chapter 3 - Data Validation and Standardization
