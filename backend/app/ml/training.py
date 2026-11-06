"""Reproducible candidate training and artifact validation for AI4I."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.ml.dataset import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_SENSOR_COLUMNS,
    TARGET_COLUMN,
    require_valid_dataset,
)

ARTIFACT_SCHEMA_VERSION = 1
TARGET_DEFINITION = (
    "Same-datapoint Machine failure event; not a future-horizon or remaining-useful-life "
    "target."
)
PACKAGE_NAMES = ("joblib", "numpy", "pandas", "scikit-learn", "xgboost")


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed split, metric threshold, and random seed for one training run."""

    random_seed: int = 42
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    operating_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be less than 1.")
        if not 0 < self.operating_threshold < 1:
            raise ValueError("operating_threshold must be between 0 and 1.")


class ModelArtifactError(ValueError):
    """Raised when a model artifact is incomplete, modified, or incompatible."""


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fitted only within each estimator's training split."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
            ("numeric", StandardScaler(), list(NUMERIC_SENSOR_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _build_estimators(random_seed: int, positive_weight: float) -> dict[str, Any]:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_seed,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=positive_weight,
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def _build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def _positive_class_weight(labels: pd.Series) -> float:
    counts = labels.value_counts()
    if set(counts.index) != {0, 1} or counts.get(1, 0) == 0:
        raise ValueError("Training labels must include both failure classes 0 and 1.")
    return float(counts[0] / counts[1])


def _classification_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    actual = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or len(scores) != len(actual):
        raise ValueError("Expected one failure probability per label.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Model probabilities must be finite values between 0 and 1.")
    predicted = (scores >= threshold).astype(int)
    return {
        "operating_threshold": threshold,
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(actual, scores)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "brier_score": float(brier_score_loss(actual, scores)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
    }
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "unknown"
    return versions


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    config: TrainingConfig | None = None,
    dataset_path: str | Path | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Select using validation average precision and evaluate the refit model once on test."""
    selected_config = config or TrainingConfig()
    require_valid_dataset(frame)

    class_counts = frame[TARGET_COLUMN].value_counts()
    if set(class_counts.index) != {0, 1} or int(class_counts.min()) < 10:
        raise ValueError(
            "Training requires at least 10 examples from each target class for reliable "
            "stratified train/validation/test splits."
        )

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    labels = frame[TARGET_COLUMN].astype(int)
    train_validation_x, test_x, train_validation_y, test_y = train_test_split(
        features,
        labels,
        test_size=selected_config.test_fraction,
        random_state=selected_config.random_seed,
        stratify=labels,
    )
    relative_validation_fraction = selected_config.validation_fraction / (
        1 - selected_config.test_fraction
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_validation_x,
        train_validation_y,
        test_size=relative_validation_fraction,
        random_state=selected_config.random_seed,
        stratify=train_validation_y,
    )

    estimators = _build_estimators(
        selected_config.random_seed,
        _positive_class_weight(train_y),
    )
    validation_results: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        candidate = _build_pipeline(estimator)
        candidate.fit(train_x, train_y)
        probabilities = candidate.predict_proba(validation_x)[:, 1]
        validation_results[name] = _classification_metrics(
            validation_y,
            probabilities,
            selected_config.operating_threshold,
        )

    selected_name = max(
        validation_results,
        key=lambda name: (validation_results[name]["average_precision"], name),
    )
    final_estimator = _build_estimators(
        selected_config.random_seed,
        _positive_class_weight(train_validation_y),
    )[selected_name]
    final_pipeline = _build_pipeline(final_estimator)
    final_pipeline.fit(train_validation_x, train_validation_y)
    test_probabilities = final_pipeline.predict_proba(test_x)[:, 1]
    test_metrics = _classification_metrics(
        test_y,
        test_probabilities,
        selected_config.operating_threshold,
    )

    dataset_sha256 = _sha256(Path(dataset_path)) if dataset_path is not None else None
    timestamp = datetime.now(UTC).isoformat()
    run_id = uuid4().hex
    metadata: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": f"sentinelml-ai4i-{run_id[:12]}",
        "run_id": run_id,
        "trained_at_utc": timestamp,
        "selected_model": selected_name,
        "target_column": TARGET_COLUMN,
        "target_definition": TARGET_DEFINITION,
        "feature_names": list(FEATURE_COLUMNS),
        "excluded_identifier_columns": list(ID_COLUMNS),
        "excluded_target_columns": [TARGET_COLUMN, "TWF", "HDF", "PWF", "OSF", "RNF"],
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "source": "UCI Machine Learning Repository, DOI 10.24432/C5HS5C",
            "synthetic": True,
            "sha256": dataset_sha256,
            "row_count": len(frame),
        },
        "training_config": asdict(selected_config),
        "split": {
            "strategy": "stratified train/validation/test split",
            "random_seed": selected_config.random_seed,
            "train_rows": len(train_x),
            "validation_rows": len(validation_x),
            "test_rows": len(test_x),
            "train_class_counts": {
                str(int(label)): int(count)
                for label, count in train_y.value_counts().sort_index().items()
            },
            "validation_class_counts": {
                str(int(label)): int(count)
                for label, count in validation_y.value_counts().sort_index().items()
            },
            "test_class_counts": {
                str(int(label)): int(count)
                for label, count in test_y.value_counts().sort_index().items()
            },
            "selection_metric": "validation average_precision",
        },
        "validation_metrics_by_candidate": validation_results,
        "test_metrics": test_metrics,
        "package_versions": _package_versions(),
        "limitations": [
            "Source data is synthetic and is not live factory telemetry.",
            "The target describes failure at the recorded datapoint, not a future horizon.",
            "Model outputs are experimental and not calibrated real-world failure probabilities.",
            (
                "Test metrics are a single held-out evaluation and should not be used to tune "
                "candidates."
            ),
        ],
    }
    return final_pipeline, metadata


def save_model_artifact(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    artifact_dir: str | Path,
) -> Path:
    """Persist a fitted pipeline and checksum-protected metadata to a new directory."""
    destination = Path(artifact_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Artifact directory {destination} is not empty; choose a new run directory."
        )
    destination.mkdir(parents=True, exist_ok=True)

    model_path = destination / "pipeline.joblib"
    metadata_path = destination / "metadata.json"
    model_temp: Path | None = None
    metadata_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination, delete=False, prefix=".pipeline.", suffix=".joblib"
        ) as model_file:
            model_temp = Path(model_file.name)
        joblib.dump(pipeline, model_temp)
        artifact_metadata = dict(metadata)
        artifact_metadata["pipeline_sha256"] = _sha256(model_temp)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            delete=False,
            prefix=".metadata.",
            suffix=".json",
        ) as metadata_file:
            json.dump(artifact_metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
            metadata_temp = Path(metadata_file.name)
        os.replace(model_temp, model_path)
        os.replace(metadata_temp, metadata_path)
    finally:
        for temporary_path in (model_temp, metadata_temp):
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    return destination


def load_model_artifact(artifact_dir: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load and validate a locally generated artifact; never accept an uploaded model path."""
    source = Path(artifact_dir).resolve()
    model_path = source / "pipeline.joblib"
    metadata_path = source / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise ModelArtifactError(
            "Artifact directory must contain the generated pipeline.joblib and metadata.json."
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelArtifactError(f"Cannot read model metadata: {error}") from error

    if not isinstance(metadata, dict):
        raise ModelArtifactError("Artifact metadata is not a valid JSON object.")

    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError("Unsupported or missing artifact schema version.")
    if metadata.get("feature_names") != list(FEATURE_COLUMNS):
        raise ModelArtifactError("Artifact feature names do not match the supported AI4I schema.")
    if metadata.get("target_column") != TARGET_COLUMN:
        raise ModelArtifactError("Artifact target does not match the supported target.")
    if metadata.get("target_definition") != TARGET_DEFINITION:
        raise ModelArtifactError("Artifact target definition is missing or unsupported.")
    if metadata.get("pipeline_sha256") != _sha256(model_path):
        raise ModelArtifactError("Pipeline checksum does not match the artifact metadata.")
    test_metrics = metadata.get("test_metrics")
    if not isinstance(test_metrics, dict):
        raise ModelArtifactError("Artifact is missing test evaluation metrics.")
    if not metadata.get("model_version") or not metadata.get("trained_at_utc"):
        raise ModelArtifactError("Artifact is missing model version or training timestamp.")
    metric_names = (
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "accuracy",
        "brier_score",
    )
    for metric_name in metric_names:
        value = test_metrics.get(metric_name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ModelArtifactError(f"Artifact test metric '{metric_name}' is invalid.")
    threshold = test_metrics.get("operating_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ModelArtifactError("Artifact operating threshold is invalid.")
    matrix = test_metrics.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or any(not isinstance(value, int) or value < 0 for value in row)
            for row in matrix
        )
    ):
        raise ModelArtifactError("Artifact confusion matrix is invalid.")

    try:
        pipeline = joblib.load(model_path)
    except Exception as error:
        raise ModelArtifactError(
            f"Cannot deserialize the generated model artifact: {error}"
        ) from error

    if not isinstance(pipeline, Pipeline) or "classifier" not in pipeline.named_steps:
        raise ModelArtifactError("Artifact does not contain a supported scikit-learn pipeline.")
    classifier = pipeline.named_steps["classifier"]
    if not np.array_equal(getattr(classifier, "classes_", None), np.array([0, 1])):
        raise ModelArtifactError("Artifact classifier does not expose expected classes [0, 1].")
    if not hasattr(pipeline, "predict_proba"):
        raise ModelArtifactError("Artifact classifier does not support probability estimates.")
    if list(getattr(pipeline, "feature_names_in_", ())) != list(FEATURE_COLUMNS):
        raise ModelArtifactError("Fitted pipeline input features do not match the AI4I schema.")

    probe = pd.DataFrame(
        [
            {
                "Type": "L",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 0,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    try:
        probe_probabilities = np.asarray(pipeline.predict_proba(probe), dtype=float)
    except Exception as error:
        raise ModelArtifactError(
            f"Artifact failed its inference readiness check: {error}"
        ) from error
    if (
        probe_probabilities.shape != (1, 2)
        or not np.isfinite(probe_probabilities).all()
        or (probe_probabilities < 0).any()
        or (probe_probabilities > 1).any()
    ):
        raise ModelArtifactError("Artifact failed its inference readiness check.")
    return pipeline, metadata


def predict_failure_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return finite failure-class probabilities for supported feature columns."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a scikit-learn Pipeline instance.")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if tuple(features.columns) != FEATURE_COLUMNS:
        raise ValueError(f"Prediction features must be exactly {FEATURE_COLUMNS}.")
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Pipeline does not support probability estimation.")
    probabilities = pipeline.predict_proba(features)[:, 1]
    invalid = (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    )
    if invalid:
        raise ValueError("Model returned probabilities outside the [0, 1] range.")
    return probabilities
