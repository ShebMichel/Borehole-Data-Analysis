# Chapter 1: Scoping Your Geotechnical Project

## Why Most Borehole Analysis Projects Fail

Many geotechnical data science projects never make it past the notebook stage. They analyze a single site, create some plots, and stop there. Hiring managers want to see systems that solve recurring problems across multiple sites.

## Identifying Real-World Problems

### Industry Pain Points

Geotechnical firms face these challenges daily:

1. **Manual log interpretation** - Engineers spend hours reading drilling logs
2. **Inconsistent classification** - Different geologists classify the same soil differently
3. **Limited spatial understanding** - 2D cross-sections miss critical subsurface features
4. **Slow turnaround** - Clients wait weeks for preliminary reports
5. **Cost overruns** - Unexpected soil conditions during construction

### Selecting Your Focus

Pick one problem that demonstrates end-to-end thinking:

**Good**: "Automated soil classification from SPT blow counts and visual descriptions"
**Better**: "Real-time lithology prediction API for drilling crews with uncertainty quantification"

## Defining Success Metrics

### Business Metrics

- **Time savings**: Reduce log interpretation from 4 hours to 15 minutes per borehole
- **Cost reduction**: Cut laboratory testing needs by 30% through predictive modeling
- **Accuracy improvement**: Match expert geologist classification 85%+ of the time

### Technical Metrics

- **Classification accuracy**: F1-score > 0.80 for major soil types (clay, sand, gravel)
- **Inference latency**: < 200ms per borehole prediction
- **Model confidence**: Calibrated probability scores (ECE < 0.1)

### Example Metrics Definition

```python
# src/metrics/project_kpis.py

from dataclasses import dataclass
from typing import Dict

@dataclass
class ProjectKPIs:
    """Business and technical success criteria"""
    
    # Business metrics
    target_time_savings_pct: float = 75.0  # 4 hours -> 1 hour
    target_cost_reduction_pct: float = 30.0
    
    # Technical metrics
    min_f1_score: float = 0.80
    max_inference_latency_ms: float = 200
    max_expected_calibration_error: float = 0.1
    
    def evaluate_business_impact(self, 
                                 baseline_hours: float,
                                 new_hours: float,
                                 baseline_cost: float,
                                 new_cost: float) -> Dict[str, bool]:
        """Check if business targets are met"""
        
        time_savings = 100 * (1 - new_hours / baseline_hours)
        cost_reduction = 100 * (1 - new_cost / baseline_cost)
        
        return {
            'time_target_met': time_savings >= self.target_time_savings_pct,
            'cost_target_met': cost_reduction >= self.target_cost_reduction_pct,
            'time_savings_pct': time_savings,
            'cost_reduction_pct': cost_reduction
        }
```

## System Architecture Overview

### Components

```
┌─────────────────┐
│  Field Data     │
│  (CSV/Excel)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation &   │
│  Preprocessing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Pipeline    │
│  (Classification)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  REST API       │
│  (FastAPI)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │
│  Dashboard      │
└─────────────────┘
```

### Technology Stack

```python
# requirements.txt

# Core data processing
pandas==2.1.0
numpy==1.24.3
geopandas==0.14.0

# Machine learning
scikit-learn==1.3.0
xgboost==2.0.0
imbalanced-learn==0.11.0

# Visualization
matplotlib==3.7.2
plotly==5.17.0
pyvista==0.42.0

# API and deployment
fastapi==0.103.0
uvicorn==0.23.2
pydantic==2.3.0

# Testing and quality
pytest==7.4.0
pytest-cov==4.1.0
black==23.7.0
```

## Project Scope Document

Create a one-page scope that you can show in interviews:

```markdown
# Borehole Lithology Classification System

## Problem
Manual soil classification from drilling logs takes 4+ hours per site and 
varies between geologists, leading to inconsistent foundation designs.

## Solution
Automated classification system using SPT data, visual descriptions, and 
depth information to predict soil types with confidence scores.

## Success Criteria
- F1-score ≥ 0.80 for 5 major soil classes
- API response time < 200ms
- 75% reduction in manual interpretation time
- Deployable to AWS with <$50/month operating cost

## Deliverables
1. Data validation pipeline
2. Trained classification model with evaluation report
3. REST API with Swagger documentation
4. 3D visualization dashboard
5. Docker deployment configuration
6. CI/CD pipeline with automated tests

## Timeline
- Week 1-2: Data collection and validation
- Week 3-4: Model development and evaluation
- Week 5-6: API and visualization
- Week 7-8: Deployment and documentation
```

## What Hiring Managers Look For

✅ **Clear problem statement** tied to real industry pain
✅ **Quantified success metrics** (not just "improve accuracy")
✅ **End-to-end thinking** (data → model → deployment → monitoring)
✅ **Cost awareness** (cloud resources, API limits)
✅ **Failure mode planning** (what happens with bad input data?)

❌ Vague goals like "analyze borehole data"
❌ Only Jupyter notebooks with no production code
❌ No consideration of deployment or maintenance

## Exercise

Before moving to Chapter 2, document your project scope:

1. What specific geotechnical problem are you solving?
2. What are 3 business metrics and 3 technical metrics?
3. Sketch your system architecture (5-7 components max)
4. What's your monthly cloud budget target?

---

**Next**: Chapter 2 - Acquiring Borehole Data
