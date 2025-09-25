import os
import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cloudpickle
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DEFAULT_THRESHOLDS = {
    "readmission": 0.5,
    "medication_change": 0.5,
}

RISK_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 0.34, "Low"),
    (0.34, 0.67, "Moderate"),
    (0.67, 1.01, "High"),  # upper bound slightly >1 to capture rounding artefacts
)

ADMISSION_TYPE_LOOKUP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped",
}

MEDICATION_FIELDS: Tuple[str, ...] = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)

MEDICATION_CATEGORIES = ("No", "Steady", "Up", "Down")

def diag_group_first_char(x):
    x = str(x)
    if not x or x.lower() == "nan": return "UNK"
    ch = x[0].upper()
    return ch if ch in list("0123456789VE") else "UNK"

def clean_diag_code(code: object) -> str:
    if code is None:
        return "UNK"
    code_str = str(code).strip().upper()
    return code_str if code_str else "UNK"


def fill_admission_description(values: Dict[str, object]) -> None:
    if values.get("admission_type_desc"):
        return
    try:
        admission_id = int(values.get("admission_type_id"))
    except (TypeError, ValueError):
        admission_id = None
    if admission_id in ADMISSION_TYPE_LOOKUP:
        values["admission_type_desc"] = ADMISSION_TYPE_LOOKUP[admission_id]


def preprocess_patient_input(
    raw: Dict[str, object],
    feature_order: Iterable[str],
    num_cols: Iterable[str],
    cat_cols: Iterable[str],
    train_medians: Optional[Dict[str, float]] = None,
):
    raw = dict(raw)
    fill_admission_description(raw)
    raw["diag_2"] = clean_diag_code(raw.get("diag_2"))
    raw["diag_3"] = clean_diag_code(raw.get("diag_3"))
    dfp = pd.DataFrame([raw]).copy()
    for c in ["number_outpatient","number_emergency","number_inpatient"]:
        if c not in dfp: dfp[c] = 0
    dfp["utilization_total_visits"] = (
        pd.to_numeric(dfp.get("number_outpatient",0), errors="coerce").fillna(0) +
        pd.to_numeric(dfp.get("number_emergency",0), errors="coerce").fillna(0) +
        pd.to_numeric(dfp.get("number_inpatient",0), errors="coerce").fillna(0)
    )
    dfp["diag_2_group"] = dfp.get("diag_2", "UNK").apply(diag_group_first_char)
    dfp["diag_3_group"] = dfp.get("diag_3", "UNK").apply(diag_group_first_char)
    for c in feature_order:
        if c not in dfp.columns:
            default_value = 0 if c in num_cols else "__MISSING__"
            dfp[c] = default_value
    if train_medians:
        numeric_fill_values = {c: train_medians.get(c, 0) for c in num_cols}
    else:
        numeric_fill_values = {c: 0 for c in num_cols}
    dfp = dfp[feature_order]
    for c in num_cols:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(numeric_fill_values[c])
    for c in cat_cols:
        dfp[c] = dfp[c].astype("string").fillna("__MISSING__")
    return dfp

def load_artifacts():
    with open(os.path.join(MODELS_DIR, "multi_pipeline.pkl"), "rb") as f:
        pipe = cloudpickle.load(f)
    with open(os.path.join(MODELS_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    num_cols = meta["features"]["num"]
    cat_cols = meta["features"]["cat"]
    feature_order = num_cols + cat_cols
    los_classes = meta.get("los_classes", ["short","medium","long"])
    diag_map = meta.get("diaggrp_mapping", {})
    inv_diag_map = {int(v): k for k, v in diag_map.items()}
    diag_pretty_labels = {
        "1": "Circulatory",
        "2": "Respiratory",
        "3": "Digestive",
        "4": "Diabetes/Endocrine",
        "5": "Injury/Poisoning",
        "6": "Musculoskeletal",
        "7": "Genitourinary",
        "8": "Neoplasms",
        "9": "Other",
        "E": "External Causes",
        "V": "Supplementary",
        "UNK": "Unknown",
    }
    medians_path = os.path.join(MODELS_DIR, "feature_medians.json")
    train_medians = None
    if os.path.exists(medians_path):
        with open(medians_path, "r") as f:
            train_medians = json.load(f)
    return (
        pipe,
        feature_order,
        num_cols,
        cat_cols,
        los_classes,
        inv_diag_map,
        diag_pretty_labels,
        train_medians,
    )

(
    pipe,
    FEATURE_ORDER,
    NUM_COLS,
    CAT_COLS,
    LOS_CLASSES,
    DIAG_MAP_INV,
    DIAGNOSIS_GROUP_NAMES,
    TRAIN_NUMERIC_MEDIANS,
) = load_artifacts()

def _risk_bucket(prob: float) -> str:
    for lower, upper, label in RISK_BANDS:
        if lower <= prob < upper:
            return label
    return RISK_BANDS[-1][2]


def _top_k_labels(probs: np.ndarray, labels: Sequence, k: int = 3) -> List[Tuple[object, float]]:
    k = min(k, len(labels))
    top_indices = np.argsort(probs)[::-1][:k]
    return [(labels[idx], float(probs[idx])) for idx in top_indices]


def predict_patient(
    patient_dict: Dict[str, object],
    thr: float = 0.5,
    thresholds: Optional[Dict[str, float]] = None,
    include_distributions: bool = True,
    top_k: int = 3,
):
    thresholds = thresholds or {}
    thr_readm = thresholds.get("readmission", thresholds.get("readmission_risk", thr))
    thr_med_change = thresholds.get("medication_change", thr)
    dfp = preprocess_patient_input(
        patient_dict,
        FEATURE_ORDER,
        NUM_COLS,
        CAT_COLS,
        train_medians=TRAIN_NUMERIC_MEDIANS,
    )
    all_probs = pipe.predict_proba(dfp)
    results = {}
    pr = float(all_probs[0][:,1][0])
    readmission_pred = "Yes" if pr >= thr_readm else "No"
    results["Readmission (<30d)"] = {
        "Prediction": readmission_pred,
        "Probability": round(pr * 100, 2),
        "Risk": _risk_bucket(pr),
        "Threshold": thr_readm,
    }
    los_vec = all_probs[1][0]
    los_idx = int(np.argmax(los_vec))
    los_prediction = LOS_CLASSES[los_idx]
    results["Length of Stay"] = {
        "Prediction": los_prediction,
        "Probability": round(float(np.max(los_vec)) * 100, 2),
        "Top Alternatives": [
            {"Label": label, "Probability": round(prob * 100, 2)}
            for label, prob in _top_k_labels(los_vec, LOS_CLASSES, k=top_k)
        ],
    }
    pm = float(all_probs[2][:,1][0])
    med_change_pred = "Yes" if pm >= thr_med_change else "No"
    results["Medication Change"] = {
        "Prediction": med_change_pred,
        "Probability": round(pm * 100, 2),
        "Risk": _risk_bucket(pm),
        "Threshold": thr_med_change,
    }
    dg_vec = all_probs[3][0]
    dg_idx = int(np.argmax(dg_vec))
    diag_code = DIAG_MAP_INV.get(dg_idx, str(dg_idx))
    diag_name = DIAGNOSIS_GROUP_NAMES.get(diag_code, diag_code)
    top_diag = _top_k_labels(dg_vec, list(range(len(dg_vec))), k=top_k)
    results["Diagnosis Group"] = {
        "Prediction": f"{diag_name} ({diag_code})",
        "Probability": round(float(np.max(dg_vec)) * 100, 2),
        "Top Alternatives": [
            {
                "Label": f"{DIAGNOSIS_GROUP_NAMES.get(DIAG_MAP_INV.get(int(idx), str(idx)), DIAG_MAP_INV.get(int(idx), str(idx)))} ({DIAG_MAP_INV.get(int(idx), str(idx))})",
                "Probability": round(prob * 100, 2),
            }
            for idx, prob in top_diag
        ],
    }
    if include_distributions:
        results["_distributions"] = {
            "length_of_stay": {label: float(prob) for label, prob in zip(LOS_CLASSES, los_vec)},
            "diagnosis_group": {
                DIAGNOSIS_GROUP_NAMES.get(DIAG_MAP_INV.get(idx, str(idx)), DIAG_MAP_INV.get(idx, str(idx))): float(prob)
                for idx, prob in enumerate(dg_vec)
            },
        }
    return results
