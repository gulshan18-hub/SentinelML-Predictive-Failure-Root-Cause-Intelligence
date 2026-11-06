"""Download, validate, and summarize the AI4I 2020 dataset."""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from app.ml.dataset import (
    DATASET_FILENAME,
    UCI_DATASET_URL,
    load_dataset,
    require_valid_dataset,
    write_eda_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "eda-report.md"
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024


def download_dataset(destination: Path, *, overwrite: bool = False) -> Path:
    """Download the official CSV, validate it, then atomically save it locally."""
    destination = destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass --overwrite-download to replace it."
        )

    request = Request(
        UCI_DATASET_URL,
        headers={"User-Agent": "SentinelML dataset preparation/0.1"},
    )
    with urlopen(request, timeout=30) as response:
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"Downloaded dataset exceeds {MAX_DOWNLOAD_BYTES} bytes.")

    candidate = pd.read_csv(io.BytesIO(payload))
    require_valid_dataset(candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, delete=False, prefix=f".{destination.name}."
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR / DATASET_FILENAME,
        help="CSV path; defaults to data/raw/ai4i2020.csv.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official UCI CSV to --input before validating it.",
    )
    parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Replace --input when used with --download.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Markdown EDA report destination; defaults to docs/eda-report.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download:
        source = download_dataset(args.input, overwrite=args.overwrite_download)
    else:
        source = args.input

    frame = load_dataset(source)
    report = write_eda_report(frame, args.report)
    validation = require_valid_dataset(frame)
    print(
        f"Validated {validation.row_count:,} rows from {source}. "
        f"Failure rate: {validation.failure_rate:.2%}. EDA report: {report}"
    )


if __name__ == "__main__":
    main()
