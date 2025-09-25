import os, json, cloudpickle, numpy as np, pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def diag_group_first_char(x):
    x = str(x)
    if not x or x.lower() == "nan": return "UNK"
    ch = x[0].upper()
    return ch if ch in list("0123456789VE") else "UNK"

def preprocess_patient_input(raw: dict, feature_order, num_cols, cat_cols, train_medians=None):
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
            dfp[c] = 0 if c in num_cols else "__MISSING__"
    dfp = dfp[feature_order]
    for c in num_cols:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0)
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
    return pipe, feature_order, num_cols, cat_cols, los_classes, diag_map

pipe, FEATURE_ORDER, NUM_COLS, CAT_COLS, LOS_CLASSES, DIAG_MAP = load_artifacts()

def predict_patient(patient_dict, thr=0.5):
    dfp = preprocess_patient_input(patient_dict, FEATURE_ORDER, NUM_COLS, CAT_COLS)
    all_probs = pipe.predict_proba(dfp)
    results = {}
    pr = float(all_probs[0][:,1][0])
    results["Readmission (<30d)"] = {"Prediction": "Yes" if pr>=thr else "No", "Probability": round(pr*100,2)}
    los_vec = all_probs[1][0]
    los_idx = int(np.argmax(los_vec))
    results["Length of Stay"] = {"Prediction": LOS_CLASSES[los_idx], "Probability": round(float(np.max(los_vec))*100,2)}
    pm = float(all_probs[2][:,1][0])
    results["Medication Change"] = {"Prediction": "Yes" if pm>=thr else "No", "Probability": round(pm*100,2)}
    dg_vec = all_probs[3][0]
    dg_idx = int(np.argmax(dg_vec))
    inv = {v:k for k,v in DIAG_MAP.items()}
    results["Diagnosis Group"] = {"Prediction": inv.get(dg_idx, str(dg_idx)), "Probability": round(float(np.max(dg_vec))*100,2)}
    return results
