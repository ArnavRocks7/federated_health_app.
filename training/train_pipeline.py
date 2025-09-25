"""Command line utility to (re)train the healthcare multi-task model.

The script expects a tabular dataset with the same feature space that the
Streamlit application collects from users.  It performs LightGBM-based
multi-task classification, evaluates the hold-out performance, performs a
light-weight search for discriminative thresholds and persists the artefacts in
``models/`` so that the dashboard can immediately consume the refreshed model.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import sklearn

from app.inference import (
    ADMISSION_TYPE_LOOKUP,
    MEDICATION_CATEGORIES,
    MEDICATION_FIELDS,
    clean_diag_code,
    diag_group_first_char,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

NUMERIC_FEATURES: Tuple[str, ...] = (
    "discharge_disposition_id",
    "admission_source_id",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "utilization_total_visits",
)

CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "payer_code",
    "medical_specialty",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    *MEDICATION_FIELDS,
    "diabetesMed",
    "admission_type_desc",
    "diag_2_group",
    "diag_3_group",
)

TARGET_DEFAULTS = {
    "readmission": "readmitted_30d",
    "length_of_stay": "length_of_stay",
    "medication_change": "medication_change",
    "diagnosis_group": "diagnosis_group",
}


@dataclass
class EncodedTargets:
    """Container for encoded target arrays and associated metadata."""

    matrix: pd.DataFrame
    los_classes: List[str]
    diagnosis_mapping: Dict[str, int]


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalise_binary(series: pd.Series) -> pd.Series:
    """Normalise a binary target to the {0, 1} domain."""

    if series.dtype == bool:
        return series.astype(int)

    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }
    mapped = cleaned.map(mapping)
    if mapped.isnull().any():
        raise ValueError("Binary target contains unrecognised values: " f"{series.unique()!r}")
    return mapped.astype(int)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "admission_type_desc" not in df:
        df["admission_type_desc"] = ""
    missing_desc = df["admission_type_desc"].isna() | (df["admission_type_desc"].astype(str).str.strip() == "")
    if missing_desc.any():
        df.loc[missing_desc, "admission_type_desc"] = (
            df.loc[missing_desc, "admission_type_id"].map(ADMISSION_TYPE_LOOKUP).fillna("__MISSING__")
        )
    for col in ("diag_2", "diag_3"):
        if col not in df:
            df[col] = "UNK"
        df[col] = df[col].apply(clean_diag_code)
    for col in ("number_outpatient", "number_emergency", "number_inpatient"):
        if col not in df:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["utilization_total_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["diag_2_group"] = df["diag_2"].apply(diag_group_first_char)
    df["diag_3_group"] = df["diag_3"].apply(diag_group_first_char)
    for med_col in MEDICATION_FIELDS:
        if med_col not in df:
            df[med_col] = "No"
        df[med_col] = df[med_col].where(df[med_col].isin(MEDICATION_CATEGORIES), "No")
    if "diabetesMed" in df:
        df["diabetesMed"] = df["diabetesMed"].where(df["diabetesMed"].isin(["Yes", "No"]), "No")
    else:
        df["diabetesMed"] = "No"
    if "A1Cresult" not in df:
        df["A1Cresult"] = "None"
    else:
        df["A1Cresult"] = df["A1Cresult"].fillna("None")
    if "max_glu_serum" not in df:
        df["max_glu_serum"] = "None"
    else:
        df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    return df


def _encode_targets(
    df: pd.DataFrame,
    readm_col: str,
    los_col: str,
    med_col: str,
    diag_col: str,
) -> EncodedTargets:
    readm = _normalise_binary(df[readm_col])
    med_change = _normalise_binary(df[med_col])

    los_series = df[los_col].astype(str).str.strip().str.lower()
    los_classes = sorted(los_series.unique())
    los_map = {label: idx for idx, label in enumerate(los_classes)}
    los_encoded = los_series.map(los_map).astype(int)

    diag_series = df[diag_col].astype(str).str.strip().str.upper()
    diag_mapping = {label: idx for idx, label in enumerate(sorted(diag_series.unique()))}
    diag_encoded = diag_series.map(diag_mapping).astype(int)

    matrix = pd.DataFrame(
        {
            "readmission": readm,
            "length_of_stay": los_encoded,
            "medication_change": med_change,
            "diagnosis_group": diag_encoded,
        }
    )
    return EncodedTargets(matrix=matrix, los_classes=list(los_classes), diagnosis_mapping=diag_mapping)


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    base_estimator = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    classifier = MultiOutputClassifier(base_estimator, n_jobs=-1)
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def _find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    best_thr = 0.5
    best_score = -np.inf
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (y_proba >= thr).astype(int)
        score = f1_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


def _evaluate_model(pipe: Pipeline, X_val: pd.DataFrame, y_val: EncodedTargets) -> Dict[str, float]:
    predictions = pipe.predict(X_val)
    if isinstance(predictions, list):
        predictions = np.column_stack(predictions)
    metrics: Dict[str, float] = {}
    for idx, column in enumerate(y_val.matrix.columns):
        metrics[f"{column}_accuracy"] = accuracy_score(y_val.matrix[column], predictions[:, idx])
        if column in {"readmission", "medication_change"}:
            proba = pipe.predict_proba(X_val)[idx][:, 1]
            metrics[f"{column}_roc_auc"] = roc_auc_score(y_val.matrix[column], proba)
            metrics[f"{column}_best_threshold"] = _find_optimal_threshold(
                y_val.matrix[column], proba
            )
    return metrics


def _save_feature_medians(X_train: pd.DataFrame) -> None:
    medians = X_train[list(NUMERIC_FEATURES)].median().to_dict()
    with open(os.path.join(MODELS_DIR, "feature_medians.json"), "w") as f:
        json.dump(medians, f, indent=2)


def _save_meta(
    features: Mapping[str, Sequence[str]],
    los_classes: Sequence[str],
    diag_mapping: Mapping[str, int],
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> None:
    meta = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "features": {
            "num": list(features["num"]),
            "cat": list(features["cat"]),
        },
        "los_classes": list(los_classes),
        "diaggrp_mapping": {str(k): int(v) for k, v in diag_mapping.items()},
        "metrics": metrics,
        "optimal_thresholds": thresholds,
        "sklearn_version": sklearn.__version__,
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the healthcare multi-task model.")
    parser.add_argument("--input", required=True, help="CSV file containing the training data.")
    parser.add_argument(
        "--readmission-target",
        default=TARGET_DEFAULTS["readmission"],
        help="Column name for the 30-day readmission label (binary).",
    )
    parser.add_argument(
        "--los-target",
        default=TARGET_DEFAULTS["length_of_stay"],
        help="Column name for the length of stay label (categorical).",
    )
    parser.add_argument(
        "--medication-target",
        default=TARGET_DEFAULTS["medication_change"],
        help="Column name indicating whether the medication regimen changed (binary).",
    )
    parser.add_argument(
        "--diagnosis-target",
        default=TARGET_DEFAULTS["diagnosis_group"],
        help="Column name for the diagnosis grouping label (categorical).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used for the train/validation split.",
    )
    parser.add_argument(
        "--output",
        default=MODELS_DIR,
        help="Directory where the trained artefacts will be stored.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = _prepare_features(df)
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing_features = [col for col in required_columns if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Dataset is missing required feature columns after preprocessing: "
            + ", ".join(sorted(missing_features))
        )
    targets = _encode_targets(
        df,
        readm_col=args.readmission_target,
        los_col=args.los_target,
        med_col=args.medication_target,
        diag_col=args.diagnosis_target,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        df[list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)],
        targets.matrix,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=targets.matrix[["readmission", "medication_change"]],
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = _evaluate_model(pipeline, X_val, targets)

    optimal_thresholds = {
        "readmission": metrics.get("readmission_best_threshold", 0.5),
        "medication_change": metrics.get("medication_change_best_threshold", 0.5),
    }

    _ensure_directory(args.output)
    with open(os.path.join(args.output, "multi_pipeline.pkl"), "wb") as f:
        import cloudpickle

        cloudpickle.dump(pipeline, f)

    _save_feature_medians(X_train)
    _save_meta(
        features={"num": NUMERIC_FEATURES, "cat": CATEGORICAL_FEATURES},
        los_classes=targets.los_classes,
        diag_mapping=targets.diagnosis_mapping,
        metrics={k: float(v) for k, v in metrics.items()},
        thresholds=optimal_thresholds,
    )

    report_lines = [
        "Training complete.",
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2),
        "Optimal thresholds: " + json.dumps(optimal_thresholds),
    ]
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
