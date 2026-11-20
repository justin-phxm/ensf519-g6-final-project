# ENSF 519 Group 6 — Diabetes Classification

Compare neural-network approaches to diabetes classification when clinical measurements are missing:

- **Masked-feature MLP:** trained with randomly hidden clinical measurements and an explicit missingness mask.
- **Denoising autoencoder classifier:** reconstructs hidden measurements, then predicts diabetes status from the completed feature vector.
- **Baseline MLP:** trained without masking augmentation.

## Dataset

The Kaggle notebook [Diabetes EDA 100%](https://www.kaggle.com/code/lalit7881/diabetes-eda-100) uses the dataset `datascikhan/diabetes`, file `diabetes_large_dataset.csv`. The CSV has 100,000 rows and 209 columns, including identifiers, mixed numeric/categorical predictors, and `diabetes_status` / `diabetes_type` labels. The Kaggle notebook reports `diabetes_status` as the binary target (0 = no diabetes, 1 = diabetes).

Dataset files are kept out of Git. Download the Kaggle dataset into `data/raw/` with:

```bash
uv run python scripts/download_data.py
```

`kagglehub` uses your existing Kaggle authentication when needed. Do not commit Kaggle tokens or downloaded patient-level data.

The preprocessing code drops rows with a missing target, removes IDs and likely target-derived fields (such as `diabetes_type`, `diabetes_risk_score`, prior diagnosis, diabetes medication, glucose-monitoring, and diabetes-complication fields), imputes numeric and categorical predictors, and fits transformations on the training split only. Review `LEAKAGE_COLUMNS` in `src/diabetes_g6/data.py` before changing this policy.

## Setup

Requires Python 3.11+ and `uv`. PyTorch uses MPS on supported Apple Silicon Macs, CUDA when available, or CPU otherwise.

```bash
uv sync
```

## Train

```bash
uv run diabetes-train --model mlp --epochs 5
uv run diabetes-train --model dae --epochs 5
uv run diabetes-train --model baseline --epochs 5
```

The training command makes a stratified train/test split, trains the selected model, reports recall, F1, PR-AUC, and ROC-AUC at missing-feature rates of 0%, 10%, 20%, and 40%, and saves its weights under the ignored `outputs/` directory. Adjust batch size, epochs, and seed from the command line.

## Layout

```text
data/raw/               downloaded CSV (ignored by Git)
scripts/download_data.py Kaggle dataset download helper
src/diabetes_g6/data.py  loading, leakage exclusions, split, preprocessing
src/diabetes_g6/models.py MLP and denoising-autoencoder architectures
src/diabetes_g6/train.py training and evaluation CLI
outputs/                local model checkpoints (ignored by Git)
```
