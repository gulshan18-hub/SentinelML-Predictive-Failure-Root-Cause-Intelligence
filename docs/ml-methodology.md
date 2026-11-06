# ML methodology

## Current target

The current supervised target is `Machine failure` from the AI4I 2020 dataset. It marks a failure at the same recorded datapoint. It does not define failure within a future time horizon and does not estimate remaining useful life. The AI4I source dataset is synthetic; the resulting model and metrics are not evidence of performance in a real factory.

## Feature and preprocessing policy

The current candidate features are `Type`, air temperature, process temperature, rotational speed, torque, and tool wear. `UDI` and `Product ID` are identifiers and excluded. `Machine failure` and the five component failure labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are targets/label details and excluded from inputs.

Each candidate is an sklearn pipeline containing a `ColumnTransformer` and a classifier. It one-hot encodes the categorical type and scales numeric sensors. Every preprocessor is fitted as part of its estimator on the training partition only. The final selected candidate is refitted on train plus validation before the held-out test evaluation.

## Split, selection, and evaluation

The default run uses random seed 42 and a stratified split into 60% train, 20% validation, and 20% test. Stratification preserves class proportions; this is a random row split, not an independent-machine or prospective temporal evaluation. The dataset does not provide a documented fleet identity or usable future-failure horizon, so this split must not be interpreted as prospective production validation.

The four candidates are:

1. `DummyClassifier(strategy="prior")` reference baseline.
2. Class-weighted logistic regression.
3. Balanced random forest.
4. XGBoost with training-split-derived positive-class weight.

Candidate selection uses validation average precision (the PR-AUC summary reported by sklearn's `average_precision_score`). The configured classification threshold defaults to 0.5 and is fixed before model comparison; no threshold tuning is performed on the test partition. After choosing one candidate, it is refitted on train plus validation and evaluated once on test. The artifact includes validation metrics for candidates and final test metrics for the selected refit model.

Reported metrics include precision, recall, F1, average precision, ROC-AUC, accuracy, Brier score, and the 2x2 confusion matrix at the recorded threshold. Accuracy is supplementary, not a primary success metric. False negatives correspond to missed failure events; false positives correspond to unnecessary inspections or maintenance. Because the target describes failures at the same row, these metrics describe event classification on a synthetic benchmark only.

## Imbalance handling

The baseline remains unweighted. Logistic regression uses `class_weight="balanced"` and the random forest uses `class_weight="balanced_subsample"`. XGBoost receives a negative-to-positive class ratio calculated from the relevant training partition. No oversampling is used, and validation/test examples remain unchanged.

## Artifact contents and trust

`python -m scripts.train_model` writes one new run directory containing:

- `pipeline.joblib`: fitted preprocessing and selected classifier together.
- `metadata.json`: model version, UTC training time, dataset source and SHA-256 when trained from a file, feature/target definitions, random seed and split counts, candidate validation metrics, selected-model test metrics, and package versions.

The loader checks the artifact schema, expected features and target, artifact metadata, pipeline checksum, sklearn pipeline/classifier shape, binary classes, and probability support before returning the pipeline. Joblib uses Python object deserialization: only load artifacts generated and stored locally by this project. Do not accept model artifacts from uploads, untrusted paths, or external users. The checksum detects accidental modification but is not a cryptographic signature against an attacker able to replace both files.

The recorded probability is a model score. The Brier score is included as a calibration diagnostic, but no calibration procedure or real-world calibration study is implemented, so do not present model scores as guaranteed real-world failure probabilities.

## Run the training pipeline

From the repository root in Windows PowerShell, after installing the dependencies and downloading/validating the dataset:

```powershell
Push-Location backend
python -m scripts.train_model
Pop-Location
```

To choose a deterministic output directory or change the predeclared operating threshold:

```powershell
Push-Location backend
python -m scripts.train_model --artifact-dir ..\artifacts\ai4i_run_01 --seed 42 --threshold 0.5
Pop-Location
```

The output directory must be new or empty. The command does not use the small test fixture as a fallback; absent real dataset input is an explicit error.

## Reproducibility limits

The random seed and split are recorded, but exact bit-for-bit reproducibility can vary by package version, platform, and underlying numerical libraries. Package versions are recorded in metadata. The split is row-stratified and not grouped by machine or time. The dataset itself is synthetic and is not a substitute for validation on representative, time-ordered factory telemetry.
