# Chapter 5: Deploying Geotechnical Systems

## From Notebook to Production

A trained model in a Jupyter notebook is not a product. Production systems need:
- **API endpoints** for real-time predictions
- **Containerization** for consistent environments
- **Monitoring** to detect model drift
- **Cost tracking** to stay within budget
- **Documentation** for other engineers

## Building a REST API

```python
# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Borehole Lithology Prediction API",
    description="Predict soil types from borehole measurements",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = Path("models/lithology_classifier_v1")
model = joblib.load(MODEL_PATH / "model.pkl")
scaler = joblib.load(MODEL_PATH / "scaler.pkl")
label_encoder = joblib.load(MODEL_PATH / "label_encoder.pkl")


class BoreholeSample(BaseModel):
    """Single borehole measurement"""
    depth_m: float = Field(..., ge=0, le=500, description="Depth in meters")
    spt_n: int = Field(..., ge=0, le=200, description="SPT N-value")
    moisture: float = Field(..., ge=0, le=100, description="Moisture content %")
    relative_depth: float = Field(..., ge=0, le=1)
    spt_rolling_mean: float = Field(..., ge=0, le=200)
    distance_from_origin: float = Field(..., ge=0)
    
    @validator('spt_n')
    def spt_realistic(cls, v, values):
        if v > 150:
            raise ValueError("SPT value unusually high - check measurement")
        return v


class PredictionRequest(BaseModel):
    """Batch prediction request"""
    samples: List[BoreholeSample]
    borehole_id: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    """Prediction results with confidence"""
    borehole_id: str
    predictions: List[Dict[str, any]]


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "supported_classes": label_encoder.classes_.tolist()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_lithology(request: PredictionRequest):
    """Predict lithology for borehole samples"""
    
    try:
        # Convert to feature matrix
        features = []
        for sample in request.samples:
            features.append([
                sample.depth_m,
                sample.spt_n,
                sample.moisture,
                sample.relative_depth,
                sample.spt_rolling_mean,
                sample.distance_from_origin
            ])
        
        X = np.array(features)
        X_scaled = scaler.transform(X)
        
        # Predict
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        
        # Format response
        predictions = []
        for i, (pred, proba) in enumerate(zip(y_pred, y_proba)):
            lithology = label_encoder.inverse_transform([pred])[0]
            confidence = float(proba.max())
            
            predictions.append({
                "depth_m": request.samples[i].depth_m,
                "predicted_lithology": lithology,
                "confidence": confidence,
                "probabilities": {
                    label_encoder.classes_[j]: float(proba[j])
                    for j in range(len(proba))
                }
            })
        
        return PredictionResponse(
            borehole_id=request.borehole_id,
            predictions=predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Return model metadata"""
    
    import json
    with open(MODEL_PATH / "metadata.json") as f:
        metadata = json.load(f)
    
    return {
        "features": metadata['feature_cols'],
        "classes": metadata['classes'],
        "metrics": metadata['metrics']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## API Testing

```python
# tests/test_api.py

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    """Test prediction endpoint"""
    
    payload = {
        "borehole_id": "TEST_BH001",
        "samples": [
            {
                "depth_m": 5.0,
                "spt_n": 15,
                "moisture": 20.0,
                "relative_depth": 0.25,
                "spt_rolling_mean": 14.5,
                "distance_from_origin": 50.0
            },
            {
                "depth_m": 10.0,
                "spt_n": 35,
                "moisture": 15.0,
                "relative_depth": 0.5,
                "spt_rolling_mean": 25.0,
                "distance_from_origin": 50.0
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["borehole_id"] == "TEST_BH001"
    assert len(data["predictions"]) == 2
    assert "predicted_lithology" in data["predictions"][0]
    assert "confidence" in data["predictions"][0]


def test_invalid_input():
    """Test validation errors"""
    
    payload = {
        "borehole_id": "TEST_BH001",
        "samples": [
            {
                "depth_m": -5.0,  # Invalid: negative depth
                "spt_n": 15,
                "moisture": 20.0,
                "relative_depth": 0.25,
                "spt_rolling_mean": 14.5,
                "distance_from_origin": 50.0
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
```

## Containerization

```dockerfile
# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/lithology_classifier_v1
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Build and run
docker-compose up --build

# Test API
curl http://localhost:8000/
```

## AWS Deployment

```python
# deployment/aws/deploy.py

import boto3
import json

class AWSDeployer:
    """Deploy model to AWS infrastructure"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.ecr = boto3.client('ecr', region_name=region)
        self.ecs = boto3.client('ecs', region_name=region)
    
    def push_to_ecr(self, image_name: str, tag: str = 'latest'):
        """Push Docker image to ECR"""
        
        # Create repository if not exists
        try:
            repo = self.ecr.create_repository(repositoryName=image_name)
            print(f"Created ECR repository: {image_name}")
        except self.ecr.exceptions.RepositoryAlreadyExistsException:
            print(f"Repository {image_name} already exists")
        
        # Get login credentials
        auth = self.ecr.get_authorization_token()
        token = auth['authorizationData'][0]['authorizationToken']
        endpoint = auth['authorizationData'][0]['proxyEndpoint']
        
        print(f"Push image to: {endpoint}/{image_name}:{tag}")
        
        # Return docker push command
        return f"docker push {endpoint}/{image_name}:{tag}"
    
    def deploy_to_ecs(self, cluster_name: str, service_name: str, image_uri: str):
        """Deploy to ECS Fargate"""
        
        task_definition = {
            'family': 'borehole-api',
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '256',
            'memory': '512',
            'containerDefinitions': [
                {
                    'name': 'api',
                    'image': image_uri,
                    'portMappings': [
                        {
                            'containerPort': 8000,
                            'protocol': 'tcp'
                        }
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': '/ecs/borehole-api',
                            'awslogs-region': self.region,
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }
            ]
        }
        
        response = self.ecs.register_task_definition(**task_definition)
        print(f"Registered task definition: {response['taskDefinition']['taskDefinitionArn']}")
        
        return response


# Example usage
if __name__ == '__main__':
    deployer = AWSDeployer(region='us-east-1')
    
    # Push to ECR
    push_cmd = deployer.push_to_ecr('borehole-lithology-api')
    print(f"Run: {push_cmd}")
    
    # Deploy to ECS
    # deployer.deploy_to_ecs('my-cluster', 'borehole-api', 'image-uri')
```

## Monitoring and Logging

```python
# src/api/monitoring.py

from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Request
import time
import logging

# Metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['borehole_id', 'status']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)

confidence_histogram = Histogram(
    'prediction_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringMiddleware:
    """Track API metrics"""
    
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {latency:.3f}s")
        
        return response


def log_prediction(borehole_id: str, predictions: List[Dict], status: str = 'success'):
    """Log prediction details"""
    
    prediction_counter.labels(borehole_id=borehole_id, status=status).inc()
    
    for pred in predictions:
        confidence_histogram.observe(pred['confidence'])
        
        if pred['confidence'] < 0.7:
            logger.warning(
                f"Low confidence prediction: {borehole_id} at {pred['depth_m']}m - "
                f"{pred['predicted_lithology']} ({pred['confidence']:.2f})"
            )
```

## Cost Optimization

```python
# deployment/cost_calculator.py

class AWSCostCalculator:
    """Estimate monthly AWS costs"""
    
    # Pricing (us-east-1, as of 2026)
    FARGATE_CPU_HOUR = 0.04048  # per vCPU
    FARGATE_MEMORY_HOUR = 0.004445  # per GB
    ALB_HOUR = 0.0225
    ALB_LCU = 0.008  # per LCU-hour
    
    def estimate_monthly_cost(self, 
                             cpu_units: int = 256,
                             memory_mb: int = 512,
                             requests_per_day: int = 1000):
        """Calculate estimated monthly cost"""
        
        # Fargate compute (assuming 24/7 operation)
        cpu_cost = (cpu_units / 1024) * self.FARGATE_CPU_HOUR * 24 * 30
        memory_cost = (memory_mb / 1024) * self.FARGATE_MEMORY_HOUR * 24 * 30
        
        # Load balancer
        alb_cost = self.ALB_HOUR * 24 * 30
        
        # LCU (simplified: 1 LCU per 1000 requests/hour)
        lcu_hours = (requests_per_day / 1000 / 24) * 24 * 30
        lcu_cost = lcu_hours * self.ALB_LCU
        
        total = cpu_cost + memory_cost + alb_cost + lcu_cost
        
        return {
            'fargate_cpu': cpu_cost,
            'fargate_memory': memory_cost,
            'load_balancer': alb_cost,
            'lcu': lcu_cost,
            'total_monthly': total
        }


if __name__ == '__main__':
    calc = AWSCostCalculator()
    
    # Low traffic scenario
    costs = calc.estimate_monthly_cost(
        cpu_units=256,
        memory_mb=512,
        requests_per_day=500
    )
    
    print("=== Estimated Monthly Costs ===")
    for item, cost in costs.items():
        print(f"{item}: ${cost:.2f}")
```

## CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml

name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/borehole-api:$IMAGE_TAG .
          docker push $ECR_REGISTRY/borehole-api:$IMAGE_TAG
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service --cluster prod --service borehole-api --force-new-deployment
```

## What Interviewers Look For

✅ **Production-ready API**: FastAPI with validation and error handling
✅ **Containerization**: Docker for reproducible deployments
✅ **Cloud deployment**: Understanding of AWS/GCP services
✅ **Monitoring**: Logging and metrics collection
✅ **Cost awareness**: Estimating and optimizing cloud costs
✅ **CI/CD**: Automated testing and deployment

❌ Only local deployment
❌ No error handling or validation
❌ No monitoring or logging
❌ Ignoring costs

## Exercise

1. Build and test the FastAPI application locally
2. Containerize your application with Docker
3. Estimate monthly AWS costs for 1000 requests/day
4. Set up basic monitoring with logging
5. Document your deployment process

---

**Next**: Chapter 6 - Presenting Results to Stakeholders
