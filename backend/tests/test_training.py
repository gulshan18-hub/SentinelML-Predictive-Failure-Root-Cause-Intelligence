from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.ml.dataset import (
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
)
from app.ml.training import (
    ModelArtifactError,
    TrainingConfig,
    load_model_artifact,
    predict_failure_probability,
    save_model_artifact,
    train_and_evaluate,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "fixtures" / "ai4i_sample.csv"
)


def _synthetic_training_frame(rows: int = 240) -> pd.DataFrame:
    """Create a deterministic, test-only balanced frame with consistent failure labels."""
    rng = np.random.default_rng(120)
    target = np.tile([0, 1], rows // 2)
    rng.shuffle(target)
    types = rng.choice(["L", "M", "H"], size=rows)
    air_temperature = rng.normal(300, 1.5, size=rows)
    process_temperature = air_temperature + rng.normal(10, 0.7, size=rows)
    speed = np.maximum(rng.normal(1500, 150, size=rows), 500)
    torque = rng.normal(40, 8, size=rows).clip(min=1)
    tool_wear = rng.uniform(0, 200, size=rows)
    failure_mode = target.astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, rows + 1),
            "Product ID": [f"{kind}{index:05d}" for index, kind in enumerate(types)],
            "Type": types,
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            TARGET_COLUMN: target,
            "TWF": failure_mode,
            "HDF": np.zeros(rows, dtype=int),
            "PWF": np.zeros(rows, dtype=int),
            "OSF": np.zeros(rows, dtype=int),
            "RNF": np.zeros(rows, dtype=int),
        },
        columns=EXPECTED_COLUMNS,
    )


def test_training_refits_preprocessing_and_evaluates_test_once() -> None:
    pipeline, metadata = train_and_evaluate(
        _synthetic_training_frame(),
        config=TrainingConfig(random_seed=17),
    )

    assert isinstance(pipeline, Pipeline)
    assert metadata["split"]["train_rows"] == 144
    assert metadata["split"]["validation_rows"] == 48
    assert metadata["split"]["test_rows"] == 48
    assert metadata["split"]["selection_metric"] == "validation average_precision"
    assert metadata["selected_model"] in metadata["validation_metrics_by_candidate"]
    assert list(pipeline.feature_names_in_) == list(FEATURE_COLUMNS)
    assert "UDI" not in pipeline.feature_names_in_
    assert "Product ID" not in pipeline.feature_names_in_
    assert set(metadata["validation_metrics_by_candidate"]) == {
        "dummy_prior",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    metrics = metadata["test_metrics"]
    assert metrics["operating_threshold"] == 0.5
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert metadata["dataset"]["synthetic"] is True
    assert "future-horizon" in metadata["target_definition"]


def test_training_rejects_too_small_or_single_class_data() -> None:
    small_frame = _synthetic_training_frame(rows=6)

    with pytest.raises(ValueError, match="at least 10 examples"):
        train_and_evaluate(small_frame)


def test_training_config_rejects_invalid_split_and_threshold() -> None:
    with pytest.raises(ValueError, match="less than 1"):
        TrainingConfig(validation_fraction=0.5, test_fraction=0.5)
    with pytest.raises(ValueError, match="operating_threshold"):
        TrainingConfig(operating_threshold=1.0)


def test_model_artifact_round_trip_and_checksum_validation(tmp_path: Path) -> None:
    pipeline, metadata = train_and_evaluate(_synthetic_training_frame())
    artifact_dir = save_model_artifact(pipeline, metadata, tmp_path / "model")

    loaded_pipeline, loaded_metadata = load_model_artifact(artifact_dir)
    input_frame = pd.read_csv(FIXTURE_PATH).loc[:, FEATURE_COLUMNS].head(2)
    probabilities = predict_failure_probability(loaded_pipeline, input_frame)

    assert loaded_metadata["model_version"] == metadata["model_version"]
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    with (artifact_dir / "pipeline.joblib").open("ab") as model_file:
        model_file.write(b"tampered")
    with pytest.raises(ModelArtifactError, match="checksum"):
        load_model_artifact(artifact_dir)


def test_model_artifact_rejects_unexpected_feature_metadata(tmp_path: Path) -> None:
    pipeline, metadata = train_and_evaluate(_synthetic_training_frame())
    artifact_dir = save_model_artifact(pipeline, metadata, tmp_path / "model")
    metadata_path = artifact_dir / "metadata.json"
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_metadata["feature_names"] = ["UDI"]
    metadata_path.write_text(json.dumps(saved_metadata), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="feature names"):
        load_model_artifact(artifact_dir)


def test_missing_dataset_is_not_replaced_with_test_fixture(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Download the official dataset"):
        load_dataset(tmp_path / "absent.csv")


def test_prediction_rejects_wrong_feature_names() -> None:
    with pytest.raises(ValueError, match="exactly"):
        predict_failure_probability(Pipeline(steps=[]), pd.DataFrame({"UDI": [1]}))
