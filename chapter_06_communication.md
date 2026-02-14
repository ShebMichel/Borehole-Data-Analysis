# Chapter 6: Presenting Results to Stakeholders

## Why Communication Matters

Technical excellence means nothing if you can't explain:
- **To executives**: Business value and ROI
- **To engineers**: System architecture and maintenance
- **To domain experts**: Model limitations and failure modes
- **To clients**: Practical usage and interpretation

## Interactive Visualization Dashboard

```python
# src/visualization/dashboard.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class BoreholeDashboard:
    """Create interactive visualizations for stakeholders"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def plot_3d_subsurface(self, borehole_ids: List[str] = None):
        """3D visualization of subsurface lithology"""
        
        if borehole_ids:
            df_plot = self.df[self.df['borehole_id'].isin(borehole_ids)]
        else:
            df_plot = self.df
        
        # Color mapping for lithology
        color_map = {
            'CLAY': 'brown',
            'SAND': 'yellow',
            'GRAVEL': 'gray',
            'SILT': 'tan',
            'ROCK': 'darkgray'
        }
        
        df_plot['color'] = df_plot['lithology'].map(color_map)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=df_plot['x_coord'],
            y=df_plot['y_coord'],
            z=-df_plot['depth_m'],  # Negative for depth below surface
            mode='markers',
            marker=dict(
                size=5,
                color=df_plot['color'],
                opacity=0.8
            ),
            text=df_plot.apply(
                lambda row: f"BH: {row['borehole_id']}<br>"
                           f"Depth: {row['depth_m']:.1f}m<br>"
                           f"Lithology: {row['lithology']}<br>"
                           f"SPT: {row['spt_n']}",
                axis=1
            ),
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title='3D Subsurface Model',
            scene=dict(
                xaxis_title='X Coordinate (m)',
                yaxis_title='Y Coordinate (m)',
                zaxis_title='Depth (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            width=900,
            height=700
        )
        
        return fig
    
    def plot_cross_section(self, x_coord: float, tolerance: float = 10):
        """2D cross-section at specified X coordinate"""
        
        df_section = self.df[
            (self.df['x_coord'] >= x_coord - tolerance) &
            (self.df['x_coord'] <= x_coord + tolerance)
        ].sort_values(['y_coord', 'depth_m'])
        
        fig = px.scatter(
            df_section,
            x='y_coord',
            y='depth_m',
            color='lithology',
            title=f'Cross-Section at X = {x_coord}m',
            labels={'y_coord': 'Y Coordinate (m)', 'depth_m': 'Depth (m)'},
            hover_data=['borehole_id', 'spt_n']
        )
        
        fig.update_yaxis(autorange='reversed')
        fig.update_layout(width=1000, height=600)
        
        return fig
    
    def plot_model_performance(self, metrics: Dict):
        """Visualize model performance metrics"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'F1-Score by Class',
                'Confusion Matrix',
                'Prediction Confidence',
                'Model Comparison'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'heatmap'}],
                [{'type': 'histogram'}, {'type': 'bar'}]
            ]
        )
        
        # F1-scores
        report = metrics['classification_report']
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [report[c]['f1-score'] for c in classes]
        
        fig.add_trace(
            go.Bar(x=classes, y=f1_scores, name='F1-Score'),
            row=1, col=1
        )
        
        # Confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        fig.add_trace(
            go.Heatmap(z=cm, x=classes, y=classes, colorscale='Blues'),
            row=1, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Performance Dashboard")
        
        return fig
    
    def create_executive_summary(self, metrics: Dict, cost_estimate: Dict):
        """Generate executive summary visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Accuracy',
                'Time Savings',
                'Monthly Cost Breakdown',
                'ROI Projection'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'pie'}, {'type': 'bar'}]
            ]
        )
        
        # Accuracy indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics['test_accuracy'] * 100,
                title={'text': "Model Accuracy (%)"},
                delta={'reference': 80},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75,
                           'value': 80
                       }}
            ),
            row=1, col=1
        )
        
        # Time savings indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=75,
                title={'text': "Time Savings (%)"},
                delta={'reference': 50, 'relative': False}
            ),
            row=1, col=2
        )
        
        # Cost breakdown
        fig.add_trace(
            go.Pie(
                labels=list(cost_estimate.keys()),
                values=list(cost_estimate.values()),
                hole=0.3
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=700, showlegend=True, title_text="Executive Summary")
        
        return fig


# Example usage
if __name__ == '__main__':
    df = pd.read_csv('data/processed/site_001_features.csv')
    
    dashboard = BoreholeDashboard(df)
    
    # Generate visualizations
    fig_3d = dashboard.plot_3d_subsurface()
    fig_3d.write_html('reports/3d_subsurface.html')
    
    fig_section = dashboard.plot_cross_section(x_coord=100)
    fig_section.write_html('reports/cross_section.html')
```

## Technical Documentation

```markdown
# Technical Documentation Template

## System Overview

### Purpose
Automated lithology classification system for borehole exploration data, reducing 
manual interpretation time by 75% while maintaining 85%+ accuracy.

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Data      │────▶│  Validation  │────▶│   Feature   │
│  Ingestion  │     │  & Cleaning  │     │ Engineering │
└─────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │◀────│   REST API   │◀────│  ML Model   │
│ Application │     │  (FastAPI)   │     │  (XGBoost)  │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Technology Stack
- **ML Framework**: scikit-learn, XGBoost
- **API**: FastAPI, Pydantic
- **Deployment**: Docker, AWS ECS Fargate
- **Monitoring**: CloudWatch, Prometheus
- **Storage**: S3 for model artifacts

## Model Details

### Training Data
- **Source**: British Geological Survey + synthetic data
- **Size**: 15,000 samples from 150 boreholes
- **Features**: 6 engineered features (depth, SPT, moisture, spatial)
- **Classes**: 5 lithology types (CLAY, SAND, GRAVEL, SILT, ROCK)

### Model Performance
- **Algorithm**: XGBoost Classifier
- **F1-Score**: 0.87 (macro average)
- **Inference Time**: 45ms per borehole (avg 20 samples)
- **Confidence Calibration**: ECE = 0.08

### Known Limitations
1. **Rare lithologies**: Performance degrades for uncommon soil types (<5% of data)
2. **Depth extrapolation**: Less reliable beyond 50m depth
3. **Regional bias**: Trained primarily on UK data
4. **Missing features**: Does not use grain size distribution or Atterberg limits

## API Usage

### Authentication
```bash
export API_KEY="your-api-key"
```

### Example Request
```python
import requests

payload = {
    "borehole_id": "SITE_A_BH001",
    "samples": [
        {
            "depth_m": 5.0,
            "spt_n": 15,
            "moisture": 20.0,
            "relative_depth": 0.25,
            "spt_rolling_mean": 14.5,
            "distance_from_origin": 50.0
        }
    ]
}

response = requests.post(
    "https://api.example.com/predict",
    json=payload,
    headers={"Authorization": f"Bearer {API_KEY}"}
)

print(response.json())
```

### Response Format
```json
{
  "borehole_id": "SITE_A_BH001",
  "predictions": [
    {
      "depth_m": 5.0,
      "predicted_lithology": "SAND",
      "confidence": 0.89,
      "probabilities": {
        "CLAY": 0.05,
        "SAND": 0.89,
        "GRAVEL": 0.03,
        "SILT": 0.02,
        "ROCK": 0.01
      }
    }
  ]
}
```

## Deployment

### Local Development
```bash
docker-compose up
```

### Production Deployment
```bash
# Build and push to ECR
docker build -t borehole-api .
docker tag borehole-api:latest <ecr-uri>/borehole-api:latest
docker push <ecr-uri>/borehole-api:latest

# Deploy to ECS
aws ecs update-service --cluster prod --service borehole-api --force-new-deployment
```

### Monitoring
- **Metrics**: CloudWatch dashboard at `/dashboards/borehole-api`
- **Logs**: CloudWatch Logs group `/ecs/borehole-api`
- **Alerts**: SNS topic for prediction errors and high latency

## Maintenance

### Model Retraining
Retrain when:
- Prediction confidence drops below 0.7 for >10% of requests
- New geological regions are added
- Quarterly scheduled retraining

### Data Drift Detection
Monitor monthly:
- Feature distribution shifts
- Class balance changes
- Prediction confidence trends

## Cost Analysis

### Monthly Operating Costs (1000 requests/day)
- Fargate compute: $15
- Load balancer: $17
- Data transfer: $2
- **Total: ~$34/month**

### Cost per Prediction
- $0.001 per borehole prediction
- 95% cheaper than manual interpretation ($2/borehole)

## Support

### Contact
- Technical issues: tech-support@example.com
- Model questions: data-science@example.com
- Documentation: https://docs.example.com/borehole-api
```

## Stakeholder Presentation Template

```python
# src/reporting/presentation.py

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ProjectSummary:
    """Structure for stakeholder presentations"""
    
    # Business metrics
    time_savings_pct: float
    cost_reduction_pct: float
    accuracy_pct: float
    
    # Technical metrics
    f1_score: float
    inference_latency_ms: float
    monthly_cost_usd: float
    
    # Deployment info
    uptime_pct: float
    requests_per_day: int
    
    def generate_executive_summary(self) -> str:
        """One-page summary for executives"""
        
        return f"""
# Borehole Lithology Classification System - Executive Summary

## Business Impact
- **Time Savings**: {self.time_savings_pct:.0f}% reduction in manual interpretation time
- **Cost Reduction**: {self.cost_reduction_pct:.0f}% decrease in laboratory testing needs
- **Accuracy**: {self.accuracy_pct:.0f}% match with expert geologist classifications

## Technical Performance
- **Model Quality**: F1-score of {self.f1_score:.2f} (target: 0.80)
- **Speed**: {self.inference_latency_ms:.0f}ms average response time
- **Reliability**: {self.uptime_pct:.1f}% uptime over last 30 days

## Financial
- **Operating Cost**: ${self.monthly_cost_usd:.0f}/month
- **Cost per Prediction**: ${self.monthly_cost_usd / (self.requests_per_day * 30):.3f}
- **ROI**: 95% cost savings vs manual interpretation

## Next Steps
1. Expand to additional geological regions
2. Integrate with existing drilling management software
3. Add real-time prediction during drilling operations
"""
    
    def generate_technical_brief(self) -> str:
        """Technical summary for engineers"""
        
        return f"""
# Technical Brief - Borehole Classification System

## System Architecture
- **Model**: XGBoost multi-class classifier
- **API**: FastAPI with Pydantic validation
- **Deployment**: AWS ECS Fargate + ALB
- **Monitoring**: CloudWatch + Prometheus

## Performance Metrics
- F1-Score: {self.f1_score:.3f}
- Latency (p95): {self.inference_latency_ms * 1.2:.0f}ms
- Throughput: {self.requests_per_day} requests/day

## Infrastructure
- Compute: 0.25 vCPU, 512MB RAM
- Scaling: Auto-scale 1-3 tasks based on CPU
- Cost: ${self.monthly_cost_usd:.2f}/month

## Maintenance
- Model version: v1.0.0
- Last retrained: 2026-02-01
- Next scheduled retrain: 2026-05-01
"""


# Example usage
if __name__ == '__main__':
    summary = ProjectSummary(
        time_savings_pct=75,
        cost_reduction_pct=30,
        accuracy_pct=87,
        f1_score=0.87,
        inference_latency_ms=45,
        monthly_cost_usd=34,
        uptime_pct=99.8,
        requests_per_day=1000
    )
    
    print(summary.generate_executive_summary())
    print("\n" + "="*80 + "\n")
    print(summary.generate_technical_brief())
```

## Portfolio README Template

```markdown
# Borehole Lithology Classification System

[![Tests](https://github.com/username/borehole-analysis/workflows/tests/badge.svg)](https://github.com/username/borehole-analysis/actions)
[![Coverage](https://codecov.io/gh/username/borehole-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/username/borehole-analysis)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Production-ready ML system for automated soil classification from borehole exploration data.

## 🎯 Problem Statement

Geotechnical engineers spend 4+ hours manually interpreting each borehole log, leading to:
- Slow project turnaround times
- Inconsistent classifications between geologists
- High costs for preliminary site investigations

## 💡 Solution

Automated classification system using XGBoost that:
- Reduces interpretation time by 75%
- Achieves 87% F1-score across 5 soil types
- Provides calibrated confidence scores
- Deploys as REST API for <$35/month

## 🏗️ Architecture

[Include architecture diagram]

## 📊 Results

| Metric | Target | Achieved |
|--------|--------|----------|
| F1-Score | ≥0.80 | 0.87 |
| Latency | <200ms | 45ms |
| Cost | <$50/mo | $34/mo |
| Time Savings | 75% | 78% |

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/username/borehole-analysis
cd borehole-analysis
pip install -r requirements.txt

# Run demo
python src/main.py --demo

# Start API
docker-compose up
```

## 📁 Project Structure

```
├── data/                   # Sample datasets
├── notebooks/             # EDA and experiments
├── src/
│   ├── preprocessing/     # Data cleaning
│   ├── models/           # Training pipeline
│   ├── api/              # FastAPI application
│   └── visualization/    # Dashboards
├── tests/                # Unit tests (85% coverage)
├── deployment/           # AWS infrastructure
└── docs/                 # Technical documentation
```

## 🧪 Testing

```bash
pytest tests/ --cov=src
```

## 📈 Model Performance

[Include confusion matrix and ROC curves]

## 💰 Cost Analysis

Monthly AWS costs for 1000 requests/day:
- Fargate: $15
- Load Balancer: $17
- Data Transfer: $2
- **Total: $34**

## 🔄 CI/CD

Automated pipeline with GitHub Actions:
- Unit tests on every PR
- Docker build and push to ECR
- Deploy to ECS on merge to main

## 📚 Documentation

- [Technical Documentation](docs/technical.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

This is a portfolio project, but feedback is welcome! Open an issue or PR.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Your Name**
- Portfolio: [yourwebsite.com](https://yourwebsite.com)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com
```

## What Interviewers Look For

✅ **Clear communication**: Explaining technical concepts to non-technical audiences
✅ **Visual storytelling**: Using charts and dashboards effectively
✅ **Business focus**: Connecting technical work to business value
✅ **Documentation**: Comprehensive README and technical docs
✅ **Professionalism**: Polished presentation materials

❌ Only showing code without context
❌ No business metrics or ROI analysis
❌ Poor documentation
❌ Ignoring stakeholder needs

## Exercise

1. Create an interactive 3D visualization of your borehole data
2. Write a one-page executive summary with business metrics
3. Generate a comprehensive README for your GitHub repository
4. Create a technical documentation page
5. Prepare a 5-minute presentation explaining your project

---

## Conclusion

You now have a complete blueprint for building production-ready geotechnical data systems. Remember:

1. **Start with the problem**, not the technology
2. **Define success metrics** before writing code
3. **Build end-to-end**, from data to deployment
4. **Document everything** for future maintainers
5. **Communicate value** to different audiences

This portfolio project demonstrates you can ship real systems, not just train models.

**Good luck with your interviews!**
