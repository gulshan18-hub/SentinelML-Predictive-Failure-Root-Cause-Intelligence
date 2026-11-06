from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.ml.dataset import (
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    require_valid_dataset,
    validate_dataset,
    write_eda_report,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "fixtures" / "ai4i_sample.csv"
)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH)


def test_fixture_has_expected_schema_and_valid_labels(sample_frame: pd.DataFrame) -> None:
    assert tuple(sample_frame.columns) == EXPECTED_COLUMNS
    report = require_valid_dataset(sample_frame)
    assert report.row_count == 6
    assert report.failure_count == 2


def test_validation_rejects_missing_values(sample_frame: pd.DataFrame) -> None:
    sample_frame.loc[0, "Torque [Nm]"] = float("nan")

    report = validate_dataset(sample_frame)

    assert not report.is_valid
    assert report.missing_value_count == 1
    assert any("missing values" in error for error in report.errors)


def test_validation_rejects_duplicate_identifiers(sample_frame: pd.DataFrame) -> None:
    sample_frame.loc[1, "UDI"] = sample_frame.loc[0, "UDI"]

    report = validate_dataset(sample_frame)

    assert not report.is_valid
    assert report.duplicate_uid_count == 1


def test_validation_rejects_out_of_range_sensors_and_inconsistent_target(
    sample_frame: pd.DataFrame,
) -> None:
    sample_frame.loc[0, "Torque [Nm]"] = -1
    sample_frame.loc[1, TARGET_COLUMN] = 1

    report = validate_dataset(sample_frame)

    assert not report.is_valid
    assert any("non-negative" in error for error in report.errors)
    assert any("failure-mode label" in error for error in report.errors)


def test_validation_rejects_unexpected_columns(sample_frame: pd.DataFrame) -> None:
    sample_frame["future_failure"] = 0

    report = validate_dataset(sample_frame)

    assert not report.is_valid
    assert any("Unexpected columns" in error for error in report.errors)


def test_validation_rejects_invalid_product_id(sample_frame: pd.DataFrame) -> None:
    sample_frame.loc[0, "Product ID"] = "X10001"

    report = validate_dataset(sample_frame)

    assert not report.is_valid
    assert any("Product ID values" in error for error in report.errors)


def test_load_dataset_reports_missing_real_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Download the official dataset"):
        load_dataset(tmp_path / "not-present.csv")


def test_eda_report_uses_dataset_counts_and_excludes_leakage_features(
    sample_frame: pd.DataFrame, tmp_path: Path
) -> None:
    report_path = write_eda_report(sample_frame, tmp_path / "eda.md")
    report_text = report_path.read_text(encoding="utf-8")

    assert "Rows: 6" in report_text
    assert "Positive examples: 2" in report_text
    assert all(feature in report_text for feature in FEATURE_COLUMNS)
    assert "UDI, Product ID" in report_text
    assert "Machine failure, TWF, HDF, PWF, OSF, RNF" in report_text
