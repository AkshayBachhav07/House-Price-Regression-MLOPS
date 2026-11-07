1. System Overview & Technical ArchitectureReal estate valuation requires modeling continuous pricing functions rather than discrete classes. This project implements an automated, containerized regression pipeline predicting property prices based on structural specifications and geographic location tiers:Machine Learning Pipeline (train.py): Uses a GradientBoostingRegressor within a Scikit-Learn Pipeline combined with StandardScaler. It models non-linear interactions across square footage, bedroom/bathroom ratios, property age depreciation, and location multipliers, evaluating performance via $R^2$ score and Mean Absolute Error (MAE).REST Microservice (app.py): An asynchronous FastAPI application exposing /health and /predict endpoints. Numerical boundaries and types are strictly enforced via Pydantic v2 schemas to ensure input integrity before scoring.Automated QA Suite (test_app.py): Pytest suite validating model artifact existence, API operational status, expected regression output formats, and HTTP 422 error handling for invalid input data.Containerization (Dockerfile): Multi-stage Docker container built from python:3.11-slim, exposing port 8000.Continuous Integration & Delivery (.github/workflows/ci-cd.yml): GitHub Actions runner that automatically validates dependencies, runs headless model training, executes unit tests, and verifies the Docker build on every push to main.2.
2. Complete Project Directory Structure

3. House-Price-Regression-MLOPS/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow (test suite & Docker verification)
│
├── data/
│   └── dataset.csv            # Simulated real estate valuation dataset
│
├── models/
│   └── model.pkl              # Serialized Scikit-learn Pipeline (StandardScaler + GradientBoostingRegressor)
│
├── app.py                     # FastAPI server handling /health and /predict endpoints
├── train.py                   # Data synthesizer, regression trainer, and model exporter
├── test_app.py                # Automated Pytest suite for API endpoints and model validation
├── requirements.txt           # Production and test dependencies
├── Dockerfile                 # Container build configuration
├── .gitignore                 # Cache and virtual environment ignores
└── README.md                  # Project overview, tech stack, and API documentation


3. Complete README.md Content
You can add this documentation to your repository by running the following block directly in your terminal:

PowerShell
Set-Content -Path "README.md" -Value @'
# House Price Prediction System (End-to-End MLOps)

An end-to-end Machine Learning Operations (MLOps) regression microservice that estimates property valuations based on physical features (square footage, bedrooms, bathrooms, structure age, garage capacity) and location tiers. Built with Scikit-Learn, served with FastAPI, containerized using Docker, validated with Pytest, and automated via GitHub Actions CI/CD.

---

## 📌 Repository Architecture

```text
House-Price-Regression-MLOPS/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Automated CI/CD (lint, train, test, Docker build)
├── data/
│   └── dataset.csv            # Real estate valuation dataset
├── models/
│   └── model.pkl              # Serialized GradientBoostingRegressor Pipeline
├── app.py                     # FastAPI REST API with Pydantic validation
├── train.py                   # Data ingestion, preprocessing, regression training & export
├── test_app.py                # Pytest regression suite for API & model logic
├── requirements.txt           # Pinned production and test dependencies
├── Dockerfile                 # Container build specification
└── .gitignore                 # Environment and cache ignores
🛠️ Tech Stack
Language: Python 3.11

Machine Learning: Scikit-learn, Pandas, NumPy, Joblib

API Framework: FastAPI, Uvicorn, Pydantic

Testing: Pytest, HTTPX

DevOps & Cloud: Docker, GitHub Actions, Render

🚀 Local Setup & Installation
1. Clone the Repository
Bash
git clone [https://github.com/AkshayBachhav07/House-Price-Regression-MLOPS.git](https://github.com/AkshayBachhav07/House-Price-Regression-MLOPS.git)
cd House-Price-Regression-MLOPS
2. Create and Activate Virtual Environment
Bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Train the Model
Generates data/dataset.csv and serializes the pipeline to models/model.pkl:

Bash
python train.py
5. Run Automated Tests
Bash
pytest test_app.py -v
6. Launch the Local API Server
Bash
uvicorn app:app --reload
Access the interactive OpenAPI Swagger UI at http://127.0.0.1:8000/docs.

🐳 Docker Execution
Build Container
Bash
docker build -t house-price-api:latest .
Run Container
Bash
docker run -d -p 8000:8000 --name house-price-service house-price-api:latest
Access the containerized service at http://localhost:8000/docs.

📡 API Endpoints & Contract
1. Health Probe
Endpoint: GET /health

Response:

JSON
{
  "status": "healthy",
  "model_artifact_loaded": true
}
2. Predict House Price
Endpoint: POST /predict

Sample Request:

JSON
{
  "square_feet": 1850.0,
  "bedrooms": 3,
  "bathrooms": 2,
  "property_age_years": 4,
  "location_tier": 1,
  "garage_spaces": 2
}
Sample Response:

JSON
{
  "predicted_price": 312450.25,
  "currency": "USD",
  "status": "Success"
}
🔄 CI/CD Pipeline
The .github/workflows/ci-cd.yml workflow triggers on pushes and pull requests to main:

Checks out repository code.

Sets up Python 3.11 with pip caching.

Installs dependencies from requirements.txt.

Runs train.py to ensure artifact reproducibility.

Executes the pytest suite.

Builds the Docker container to verify deployment readiness.
'@

git add README.md
git commit -m "docs: add comprehensive project overview, architecture and API contract guide"
git push origin main
