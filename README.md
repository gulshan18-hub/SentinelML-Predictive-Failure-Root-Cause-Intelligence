# SentinelML

SentinelML is a production-style predictive maintenance platform for industrial equipment. The repository will evolve through the implementation phases described in the master prompt, starting with the project skeleton and environment setup.

## Implementation status

- Phase 1: project scaffold and environment configuration
- Phase 2: AI4I dataset ingestion, validation, and EDA pipeline
- Phase 3: baseline/candidate training, held-out evaluation, and validated model artifacts
- Later phases: anomaly detection, explanations, API, dashboard, persistence, and deployment

## Architecture

- Backend: FastAPI, Pydantic, scikit-learn, XGBoost, SHAP, MLflow, SQLite
- Frontend: React + TypeScript + Vite + Tailwind + Recharts
- Data and ML: UCI AI4I data ingestion, schema/range/label validation, EDA, training, model comparison, anomaly detection, explanations
- Deployment: Docker Compose, frontend container, backend container

## Repository layout

- `backend/` — Python API service and ML pipeline code
- `frontend/` — React app
- `data/` — raw data, processed data, fixtures
- `docs/` — project documentation
- `artifacts/` — trained model and metadata outputs
- `.github/workflows/` — CI pipelines

## Local setup (Windows PowerShell)

The project targets Python 3.12. Install Python 3.12 before setting up the environment.

```powershell
cd "C:\Users\DELL\OneDrive\Desktop\SentinelML — Predictive Failure & Root-Cause Intelligence"
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

## Dataset preparation

Download, validate, and create an EDA report from the official UCI CSV:

```powershell
Push-Location backend
python -m scripts.prepare_dataset --download
Pop-Location
```

For manual dataset placement and detailed schema, target, and limitation notes, see [docs/dataset.md](docs/dataset.md). The UCI dataset is synthetic and supports same-datapoint event classification; it does not provide a future prediction horizon or live machine telemetry. The small fixture under `data/fixtures/` is for tests only.

## Train and evaluate

The command compares the dummy baseline, logistic regression, random forest, and XGBoost. It selects using validation average precision, then evaluates the selected refit pipeline once on the untouched test partition. It refuses to use test/demo fixtures in place of the real dataset.

```powershell
Push-Location backend
python -m scripts.train_model
Pop-Location
```

The generated artifact directory contains `pipeline.joblib` and `metadata.json` with the fitted preprocessing/model, feature and target definition, dataset hash, split details, package versions, and measured metrics. See [docs/ml-methodology.md](docs/ml-methodology.md) for the split policy, metrics, imbalance handling, artifact trust requirements, and interpretation limits. Scores are not established real-world probabilities.

## Phase 2 and 3 checks

Run dataset and training tests from the backend folder:

```powershell
Push-Location backend
python -m pytest tests\test_dataset.py tests\test_training.py -q
Pop-Location
```
