"""Train AI4I model candidates and persist a validated local artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.ml.dataset import DATASET_FILENAME, load_dataset
from app.ml.training import TrainingConfig, save_model_artifact, train_and_evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / DATASET_FILENAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="Validated AI4I CSV path; defaults to data/raw/ai4i2020.csv.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="New output directory. Defaults to artifacts/ai4i_<UTC timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified splits.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Predeclared operating threshold used for classification metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir or (
        PROJECT_ROOT
        / "artifacts"
        / f"ai4i_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    frame = load_dataset(args.data)
    pipeline, metadata = train_and_evaluate(
        frame,
        config=TrainingConfig(random_seed=args.seed, operating_threshold=args.threshold),
        dataset_path=args.data,
    )
    saved_dir = save_model_artifact(pipeline, metadata, artifact_dir)
    print(
        json.dumps(
            {
                "artifact_dir": str(saved_dir),
                "model_version": metadata["model_version"],
                "selected_model": metadata["selected_model"],
                "test_metrics": metadata["test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
