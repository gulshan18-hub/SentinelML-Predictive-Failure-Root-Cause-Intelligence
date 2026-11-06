# SentinelML

SentinelML is an early-stage predictive-maintenance research project for preparing the AI4I 2020 benchmark, comparing failure-event classifiers, and saving validated model artifacts. The repository includes a React workspace dashboard and a small FastAPI health endpoint.

> **Scope:** This project does not currently ingest live machine telemetry or serve predictions. AI4I is synthetic and labels failures at the same recorded datapoint; it does not establish future failure prediction or real-factory performance.

## What is implemented

- AI4I dataset download, schema and label validation, and exploratory reporting.
- A reproducible baseline and candidate-model training workflow with stratified train/validation/test splits.
- Model selection on validation average precision and one final evaluation on the held-out test split.
- Versioned model artifacts with metadata and integrity checks.
- A FastAPI health endpoint at `/api/v1/health` and a dashboard that checks backend connectivity.
- Dockerfiles and a Docker Compose development setup.

Prediction-serving APIs, authentication, persistent run tracking, live telemetry, and production deployment are not implemented.

## Technology

- **Backend:** Python 3.12, FastAPI, pandas, scikit-learn, XGBoost
- **Frontend:** React 18, TypeScript, Vite
- **Validation:** pytest and the GitHub Actions CI workflow

## Run locally

### Backend

From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Push-Location backend
uvicorn app.main:app --reload --port 8000
```

The health endpoint is available at `http://localhost:8000/api/v1/health`.
Set `APP_CORS_ORIGINS` to a comma-separated list of trusted frontend origins if
the frontend is served from a different origin. The default allows
`http://localhost:5173`.

### Frontend

In a second PowerShell terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://localhost:5173`. To point the dashboard at another backend, set
`VITE_API_BASE_URL` before starting Vite, for example:

```powershell
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev
```

### Docker Compose

From the repository root:

```powershell
docker compose up --build
```

The development services are exposed on ports 8000 (backend) and 5173
(frontend). The Compose configuration is intended for local development, not
production deployment.

## Prepare the dataset

The download command retrieves the official CSV from the UCI Machine Learning
Repository, validates it before writing, and generates an EDA report:

```powershell
Push-Location backend
python -m scripts.prepare_dataset --download
Pop-Location
```

The official download is written to `data/raw/ai4i2020.csv`. For manual
placement, copy the CSV into that location and run
`python -m scripts.prepare_dataset` from `backend/`.

`data/fixtures/ai4i_sample.csv` is a six-row synthetic test fixture. It must not
be used as a substitute for the full dataset or as evidence of model
performance. See the [dataset guide](docs/dataset.md) for source, license,
schema, validation, and limitations.

## Train and evaluate

After preparing the full dataset:

```powershell
Push-Location backend
python -m scripts.train_model
Pop-Location
```

The workflow compares a dummy baseline, logistic regression, random forest, and
XGBoost. It selects by validation average precision, refits on train plus
validation, evaluates once on the held-out test partition, and saves a
`pipeline.joblib` and `metadata.json` under `artifacts/`.

Read the [ML methodology](docs/ml-methodology.md) for split policy, metrics,
artifact trust requirements, and interpretation limits. Only load model
artifacts generated and stored locally by this project.

## Validate changes

```powershell
Push-Location backend
python -m pytest tests -q
Pop-Location
ruff check backend
Push-Location frontend
npm run lint
npm run build
Pop-Location
```

The GitHub Actions workflow runs the backend tests and frontend production
build for pushes to `main` and pull requests.

## Repository layout

```text
backend/              FastAPI service, data pipeline, training, and tests
frontend/             React + TypeScript workspace dashboard
data/fixtures/        Small synthetic test fixture
data/raw/             Downloaded source data (not committed)
data/processed/       Generated dataset outputs (not committed)
docs/                 Dataset and ML methodology documentation
artifacts/            Generated model artifacts (not committed)
.github/workflows/    Continuous integration
```
