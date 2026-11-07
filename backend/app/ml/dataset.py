"""AI4I 2020 dataset loading, validation, and exploratory reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/601/data.csv"
DATASET_FILENAME = "ai4i2020.csv"

ID_COLUMNS = ("UDI", "Product ID")
TARGET_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
LABEL_COLUMNS = (TARGET_COLUMN,) + FAILURE_MODE_COLUMNS
NUMERIC_SENSOR_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
CATEGORICAL_FEATURE_COLUMNS = ("Type",)
FEATURE_COLUMNS = CATEGORICAL_FEATURE_COLUMNS + NUMERIC_SENSOR_COLUMNS
EXPECTED_COLUMNS = ID_COLUMNS + ("Type",) + NUMERIC_SENSOR_COLUMNS + (
    TARGET_COLUMN,
) + FAILURE_MODE_COLUMNS
NUMERIC_COLUMNS = ("UDI",) + NUMERIC_SENSOR_COLUMNS + (TARGET_COLUMN,) + FAILURE_MODE_COLUMNS


@dataclass(frozen=True)
class DatasetValidationReport:
    """Machine-readable outcome for checks performed on one dataset."""

    row_count: int
    column_count: int
    missing_value_count: int
    duplicate_row_count: int
    duplicate_uid_count: int
    failure_count: int
    failure_rate: float
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_value_count": self.missing_value_count,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_uid_count": self.duplicate_uid_count,
            "failure_count": self.failure_count,
            "failure_rate": self.failure_rate,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class DatasetValidationError(ValueError):
    """Raised when a dataset fails one or more required validation checks."""

    def __init__(self, report: DatasetValidationReport) -> None:
        self.report = report
        details = "; ".join(report.errors)
        super().__init__(f"Dataset validation failed: {details}")


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with BOM characters and surrounding column-name whitespace removed."""
    normalized = frame.copy(deep=True)
    normalized.columns = [
        str(column).replace("\ufeff", "").strip() for column in normalized.columns
    ]
    return normalized


def _format_row_examples(
    frame: pd.DataFrame, positions: np.ndarray, columns: tuple[str, ...]
) -> str:
    def display_value(value: object) -> object:
        return value.item() if isinstance(value, np.generic) else value

    examples: list[str] = []
    identifier_column = next(
        (column for column in ("UDI", "UID") if column in frame.columns), None
    )
    for position in positions[:10]:
        details = [f"CSV row {int(position) + 2}"]
        if identifier_column is not None:
            identifier = display_value(frame[identifier_column].iloc[position])
            details.append(f"{identifier_column}={identifier!r}")
        values = ", ".join(
            f"{column}={display_value(frame[column].iloc[position])!r}"
            for column in columns
        )
        examples.append(f"{', '.join(details)} ({values})")
    remaining = len(positions) - len(examples)
    if remaining:
        examples.append(f"... and {remaining} more")
    return "; ".join(examples)


def validate_dataset(frame: pd.DataFrame) -> DatasetValidationReport:
    """Check the AI4I schema, labels, missingness, uniqueness, and basic ranges."""
    errors: list[str] = []
    warnings: list[str] = []
    frame = normalize_column_names(frame)
    column_counts = pd.Series(frame.columns).value_counts()
    duplicate_columns = sorted(column_counts[column_counts > 1].index.tolist())
    if duplicate_columns:
        errors.append(
            "Column names must be unique after normalization; duplicates: "
            + ", ".join(duplicate_columns)
        )
    columns = set(frame.columns)
    expected = set(EXPECTED_COLUMNS)
    missing_columns = sorted(expected - columns)
    unexpected_columns = sorted(columns - expected)

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if unexpected_columns:
        errors.append(f"Unexpected columns: {', '.join(unexpected_columns)}")
    if frame.empty:
        errors.append("Dataset contains no rows.")

    missing_value_count = int(frame.isna().sum().sum())
    if missing_value_count:
        errors.append(f"Dataset contains {missing_value_count} missing values.")

    duplicate_row_count = int(frame.duplicated().sum()) if len(frame) else 0
    if duplicate_row_count:
        errors.append(f"Dataset contains {duplicate_row_count} duplicate rows.")

    duplicate_uid_count = 0
    if "UDI" in frame and "UDI" not in duplicate_columns:
        duplicate_uid_count = int(frame["UDI"].duplicated().sum())
        if duplicate_uid_count:
            errors.append(f"Dataset contains {duplicate_uid_count} duplicate UDI values.")

    invalid_numeric_columns: list[str] = []
    for column in NUMERIC_COLUMNS:
        if column not in frame or column in duplicate_columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            invalid_numeric_columns.append(column)
    if invalid_numeric_columns:
        errors.append(
            "Numeric columns contain invalid or infinite values: "
            + ", ".join(invalid_numeric_columns)
        )

    numeric_labels: dict[str, pd.Series] = {}
    invalid_labels: list[str] = []
    for column in LABEL_COLUMNS:
        if column not in frame or column in duplicate_columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric_labels[column] = values
        invalid_mask = values.isna() | ~values.isin((0, 1))
        if invalid_mask.any():
            invalid_labels.append(column)
            positions = np.flatnonzero(invalid_mask.to_numpy())
            errors.append(
                f"Label column '{column}' must contain only numeric 0 or 1 values; "
                f"invalid values at {_format_row_examples(frame, positions, (column,))}."
            )

    if "Type" in frame and "Type" not in duplicate_columns:
        invalid_types = sorted(set(frame["Type"].dropna().astype(str)) - {"L", "M", "H"})
        if invalid_types:
            errors.append(f"Type must be one of L, M, or H; found: {', '.join(invalid_types)}")

    if "Product ID" in frame and "Product ID" not in duplicate_columns:
        product_ids = frame["Product ID"].astype("string")
        invalid_product_ids = ~product_ids.str.fullmatch(r"[LMH]\d+", na=False)
        if invalid_product_ids.any():
            errors.append("Product ID values must start with L, M, or H and end with digits.")

    positive_columns = (
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
    )
    for column in positive_columns:
        if column in frame and column not in duplicate_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if (values.notna() & values.le(0)).any():
                errors.append(f"'{column}' must be greater than zero.")

    nonnegative_columns = ("Torque [Nm]", "Tool wear [min]")
    for column in nonnegative_columns:
        if column in frame and column not in duplicate_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if (values.notna() & values.lt(0)).any():
                errors.append(f"'{column}' must be non-negative.")

    if "UDI" in frame and "UDI" not in duplicate_columns:
        uid_values = pd.to_numeric(frame["UDI"], errors="coerce")
        invalid_uid = uid_values.dropna().le(0) | uid_values.dropna().mod(1).ne(0)
        if invalid_uid.any():
            errors.append("'UDI' must contain positive integers.")

    failure_count = 0
    failure_rate = 0.0
    if TARGET_COLUMN in numeric_labels:
        target = numeric_labels[TARGET_COLUMN]
        if target.notna().any():
            failure_count = int(target.eq(1).sum())
            failure_rate = float(target.eq(1).mean())

    if (
        all(column in numeric_labels for column in LABEL_COLUMNS)
        and not invalid_labels
        and not duplicate_columns
    ):
        target = numeric_labels[TARGET_COLUMN]
        modes = pd.DataFrame(
            {column: numeric_labels[column] for column in FAILURE_MODE_COLUMNS},
            index=frame.index,
        )
        expected_target = modes.max(axis=1)
        mismatch_positions = np.flatnonzero(target.ne(expected_target).to_numpy())
        if len(mismatch_positions):
            errors.append(
                "'Machine failure' must equal max(TWF, HDF, PWF, OSF, RNF); "
                "failure-mode label mismatch. "
                f"Found {len(mismatch_positions)} mismatched row(s): "
                f"{_format_row_examples(frame, mismatch_positions, LABEL_COLUMNS)}."
            )

    if len(frame) and failure_rate == 0.0:
        warnings.append("No positive machine-failure examples were found.")

    return DatasetValidationReport(
        row_count=len(frame),
        column_count=len(frame.columns),
        missing_value_count=missing_value_count,
        duplicate_row_count=duplicate_row_count,
        duplicate_uid_count=duplicate_uid_count,
        failure_count=failure_count,
        failure_rate=failure_rate,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(warnings),
    )


def require_valid_dataset(frame: pd.DataFrame) -> DatasetValidationReport:
    """Validate a frame and raise a detailed exception when checks fail."""
    report = validate_dataset(frame)
    if not report.is_valid:
        raise DatasetValidationError(report)
    return report


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and require the documented AI4I schema."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {source}. Download the official dataset or place "
            f"{DATASET_FILENAME} in data/raw/."
        )
    frame = normalize_column_names(pd.read_csv(source))
    require_valid_dataset(frame)
    return frame.assign(
        **{
            column: pd.to_numeric(frame[column], errors="raise")
            for column in LABEL_COLUMNS
        }
    )


def write_eda_report(frame: pd.DataFrame, report_path: str | Path) -> Path:
    """Write a Markdown EDA report using only values computed from the supplied data."""
    frame = normalize_column_names(frame)
    validation = require_valid_dataset(frame)
    frame = frame.assign(
        **{
            column: pd.to_numeric(frame[column], errors="raise")
            for column in LABEL_COLUMNS
        }
    )
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    label_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    sensor_summary = frame.loc[:, NUMERIC_SENSOR_COLUMNS].describe().round(3)
    summary_header = "| Statistic | " + " | ".join(NUMERIC_SENSOR_COLUMNS) + " |"
    summary_separator = "|---|" + "|".join("---:" for _ in NUMERIC_SENSOR_COLUMNS) + "|"
    summary_rows = [
        "| "
        + str(statistic)
        + " | "
        + " | ".join(f"{float(value):g}" for value in values)
        + " |"
        for statistic, values in sensor_summary.iterrows()
    ]
    lines = [
        "# AI4I 2020 Dataset — Exploratory Data Analysis",
        "",
        (
            "> Dataset provenance and limitations are documented in [Dataset documentation]"
            "(dataset.md)."
        ),
        "> The source data is synthetic and its labels describe a failure at the same recorded",
        "> datapoint; it does not provide a future prediction horizon or live machine stream.",
        "",
        "## Dataset overview",
        "",
        f"- Rows: {validation.row_count:,}",
        f"- Columns: {validation.column_count}",
        f"- Missing values: {validation.missing_value_count}",
        f"- Duplicate rows: {validation.duplicate_row_count}",
        f"- Duplicate UDI values: {validation.duplicate_uid_count}",
        f"- Validation: {'passed' if validation.is_valid else 'failed'}",
        "",
        "## Target distribution",
        "",
        "| Machine failure | Rows | Share |",
        "|---:|---:|---:|",
    ]
    for label in (0, 1):
        count = int(label_counts.get(label, 0))
        share = count / validation.row_count if validation.row_count else 0.0
        lines.append(f"| {label} | {count:,} | {share:.2%} |")
    lines.extend(
        [
            "",
            f"- Positive examples: {validation.failure_count:,}",
            f"- Positive rate: {validation.failure_rate:.2%}",
            "",
            "## Sensor summary",
            "",
            summary_header,
            summary_separator,
            *summary_rows,
            "",
            "## Model-input and leakage notes",
            "",
            f"- Candidate feature columns: {', '.join(FEATURE_COLUMNS)}",
            f"- Identifier columns excluded from model inputs: {', '.join(ID_COLUMNS)}",
            (
                f"- Target/failure-mode columns excluded from model inputs: "
                f"{', '.join((TARGET_COLUMN,) + FAILURE_MODE_COLUMNS)}"
            ),
            (
                "- These candidate features support same-record event classification only; "
                "a future-horizon target requires a properly time-ordered dataset and labels."
            ),
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
