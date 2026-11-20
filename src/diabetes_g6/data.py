from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "diabetes_status"
IDENTIFIER_COLUMNS = {"patient_id", "record_id"}

# These may encode diagnosis, treatment, complications, or a score derived from
# diabetes status. Keep the list explicit so the team can review it.
LEAKAGE_COLUMNS = {
    "diabetes_type", "diabetes_risk_score", "previous_diabetes_diagnosis",
    "diabetes_medication", "metformin_use", "insulin_use",
    "sulfonylurea_use", "sglt2_inhibitor_use", "glp1_medication_use",
    "dpp4_inhibitor_use", "diabetes_education_received",
    "glucose_monitoring_frequency", "home_glucose_monitoring",
    "continuous_glucose_monitor", "self_management_score",
    "diabetic_retinopathy", "diabetic_neuropathy", "diabetic_nephropathy",
    "diabetic_foot_problem", "hypoglycemia_history",
    "hyperglycemia_history", "diabetic_ketoacidosis_history",
    "cardiovascular_complication", "diabetes_complication_count",
}

# Clinical measurements to mask in the robustness experiments. Names absent
# from a dataset version are ignored.
MASKABLE_COLUMNS = {
    "height_cm", "weight_kg", "bmi", "waist_circumference_cm",
    "hip_circumference_cm", "body_fat_percentage", "lean_body_mass_kg",
    "fasting_glucose_mg_dl", "random_glucose_mg_dl",
    "post_meal_glucose_mg_dl", "two_hour_glucose_mg_dl",
    "glucose_variability", "hba1c_percent",
    "estimated_average_glucose_mg_dl", "fasting_insulin_uIU_ml",
    "c_peptide_ng_ml", "insulin_resistance_index", "systolic_bp",
    "diastolic_bp", "pulse_pressure", "mean_arterial_pressure",
    "heart_rate", "resting_heart_rate", "total_cholesterol_mg_dl",
    "ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl", "vldl_mg_dl",
    "creatinine_mg_dl", "bun_mg_dl", "egfr_ml_min_1_73m2",
    "urine_albumin_mg_g", "urine_albumin_creatinine_ratio",
    "alt_u_l", "ast_u_l", "alp_u_l", "bilirubin_mg_dl",
    "albumin_g_dl", "total_protein_g_dl", "ggt_u_l",
    "hemoglobin_g_dl", "hematocrit_percent", "rbc_count", "wbc_count",
    "platelet_count", "mcv_fl", "mch_pg", "mchc_g_dl",
    "neutrophil_percent", "lymphocyte_percent", "sodium_mmol_l",
    "potassium_mmol_l", "chloride_mmol_l", "calcium_mg_dl",
    "magnesium_mg_dl", "vitamin_d_ng_ml", "vitamin_b12_pg_ml",
    "folate_ng_ml", "uric_acid_mg_dl", "iron_ug_dl", "ferritin_ng_ml",
}


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_groups: list[np.ndarray]
    feature_names: list[str]
    dropped_leakage_columns: list[str]


def prepare_data(
    csv_path: Path, test_size: float = 0.2, random_state: int = 42
) -> PreparedData:
    """Load the CSV and fit preprocessing on the training split only."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run scripts/download_data.py first."
        )

    frame = pd.read_csv(csv_path)
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is absent.")

    y = pd.to_numeric(frame[TARGET], errors="coerce")
    labeled = y.notna() & y.isin([0, 1])
    frame = frame.loc[labeled].reset_index(drop=True)
    y = y.loc[labeled].astype(np.float32).to_numpy()

    excluded = IDENTIFIER_COLUMNS | LEAKAGE_COLUMNS | {TARGET}
    leakage_present = sorted(LEAKAGE_COLUMNS.intersection(frame.columns))
    x = frame.drop(columns=[c for c in excluded if c in frame.columns]).copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, x.nunique(dropna=True) > 1]

    if x.empty:
        raise ValueError("No usable predictor columns remain after exclusions.")

    train_idx, test_idx = train_test_split(
        np.arange(len(x)), test_size=test_size, random_state=random_state, stratify=y
    )
    x_train_frame = x.iloc[train_idx]
    x_test_frame = x.iloc[test_idx]
    numeric_columns = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [c for c in x.columns if c not in numeric_columns]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipe, numeric_columns),
         ("categorical", categorical_pipe, categorical_columns)],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    x_train = preprocessor.fit_transform(x_train_frame).astype(np.float32)
    x_test = preprocessor.transform(x_test_frame).astype(np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()

    # Numeric columns stay one-to-one; categorical columns are grouped across
    # their one-hot encodings so masks hide a whole original field at once.
    feature_groups: list[np.ndarray] = []
    for i, col in enumerate(numeric_columns):
        if col in MASKABLE_COLUMNS:
            feature_groups.append(np.asarray([i], dtype=np.int64))

    cat_offset = len(numeric_columns)
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    for col, categories in zip(categorical_columns, encoder.categories_):
        width = len(categories)
        if col in MASKABLE_COLUMNS:
            feature_groups.append(np.arange(cat_offset, cat_offset + width, dtype=np.int64))
        cat_offset += width

    if not feature_groups:
        raise ValueError("No configured clinical measurement columns were found to mask.")

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y[train_idx],
        y_test=y[test_idx],
        feature_groups=feature_groups,
        feature_names=feature_names,
        dropped_leakage_columns=leakage_present,
    )
