# Dataset, ingestion, and validation

## Source and license

SentinelML uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository (DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)). UCI describes the data as a **synthetic** dataset intended to reflect industrial predictive-maintenance data. The UCI repository lists the dataset under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Consult the linked record for current attribution and license terms when redistributing.

The official CSV endpoint used by the ingestion script is `https://archive.ics.uci.edu/static/public/601/data.csv`. Alternatively, download the CSV from the UCI record and place it at `data/raw/ai4i2020.csv`.

## Contents

The dataset contains 10,000 records and 14 columns:

| Column | Meaning | Pipeline role |
|---|---|---|
| `UDI` | Unique row identifier | ID; excluded from model inputs |
| `Product ID` | Product variant plus serial identifier | ID; excluded from model inputs |
| `Type` | Product quality variant (`L`, `M`, or `H`) | Candidate categorical feature |
| `Air temperature [K]` | Air temperature | Candidate sensor feature |
| `Process temperature [K]` | Process temperature | Candidate sensor feature |
| `Rotational speed [rpm]` | Rotational speed | Candidate sensor feature |
| `Torque [Nm]` | Torque | Candidate sensor feature |
| `Tool wear [min]` | Tool wear | Candidate sensor feature |
| `Machine failure` | Failure event label for the recorded datapoint | Target |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Tool wear, heat dissipation, power, overstrain, and random failure mode indicators | Label detail; excluded from model inputs |

The current feature candidate list consists of `Type` and the five sensor columns. Identifiers and all six failure labels are excluded from feature inputs. These definitions are centralized in `backend/app/ml/dataset.py`.

## Target semantics and limitations

`Machine failure` indicates whether a failure event occurred in the same recorded datapoint, when one or more listed failure modes are true. It is **not** a label for failure within a future time horizon, remaining useful life, or time-to-failure. The data is synthetic, not a live telemetry stream, and does not represent a fleet of identified real machines. Its measurements and results must not be presented as real factory operational evidence.

The failure-mode labels encode rules that also define the aggregate `Machine failure` target. They are deliberately excluded from model inputs to prevent direct label leakage. `UDI` and `Product ID` are identifiers and are also excluded.

## Ingestion and validation

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Push-Location backend
python -m scripts.prepare_dataset --download
Pop-Location
```

The script downloads only from the official UCI CSV endpoint, validates the downloaded file before writing it, and saves the CSV under `data/raw/`. It fails explicitly if the download, parse, or validation fails; it does not substitute the test fixture or invented data. To replace an existing download, pass `--overwrite-download`.

For a manually downloaded dataset:

```powershell
Copy-Item .\path\to\ai4i2020.csv .\data\raw\ai4i2020.csv
Push-Location backend
python -m scripts.prepare_dataset
Pop-Location
```

The validator checks the required and unexpected columns, missing and non-finite values, duplicate rows and IDs, numeric types, binary labels, label consistency, categorical product types, and basic physically invalid sensor ranges (non-positive Kelvin temperatures or rotational speed; negative torque or tool wear). It does not infer undocumented upper operating limits. A successful run writes an EDA report to `docs/eda-report.md` using statistics computed from the selected file.

## Fixture

`data/fixtures/ai4i_sample.csv` is a six-row **synthetic test fixture**. It exists only to test schema and validation code; it is not a replacement for the full UCI dataset and must not be used as a source of production performance claims.
