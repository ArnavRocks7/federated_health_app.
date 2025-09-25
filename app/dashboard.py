import json
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from inference import (
    ADMISSION_TYPE_LOOKUP,
    DEFAULT_THRESHOLDS,
    MEDICATION_CATEGORIES,
    MEDICATION_FIELDS,
    predict_patient,
)


st.set_page_config(page_title="Healthcare Dashboard", layout="wide")

RACE_OPTIONS = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
GENDER_OPTIONS = ["Female", "Male"]
AGE_OPTIONS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"]
WEIGHT_OPTIONS = [
    "None",
    "[0-25)",
    "[25-50)",
    "[50-75)",
    "[75-100)",
    "[100-125)",
    "[125-150)",
    "[150-175)",
    "[175-200)",
    "[>200]",
]
A1C_OPTIONS = ["None", "Norm", ">7", ">8"]
GLUCOSE_OPTIONS = ["None", "Norm", ">200", ">300"]
PAYER_OPTIONS = ["MC", "HM", "BC", "MD", "CM", "SP", "UN", "DM", "WC", "SI", "PO", "OT", "NO"]
DISCHARGE_OPTIONS = {
    1: "Home",
    2: "Short-term Hospital",
    3: "Skilled Nursing",
    4: "Intermediate Care",
    5: "Home Health Care",
    6: "Against Advice",
    7: "Home IV Provider",
    8: "Not Available",
    18: "Facility w/ custodial care",
    22: "Rehab Facility",
    23: "Long-term Hospital",
    24: "Nursing Facility",
    25: "Psychiatric Hospital",
}
ADMISSION_SOURCE_OPTIONS = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from Hospital",
    5: "Transfer from SNF",
    6: "Transfer from other Facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
}
MEDICAL_SPECIALTY_OPTIONS = [
    "Cardiology",
    "Endocrinology",
    "Family/General Practice",
    "InternalMedicine",
    "Nephrology",
    "Neurology",
    "Orthopedics",
    "Pulmonology",
    "Surgery-General",
]

DEFAULT_PATIENT: Dict[str, object] = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "weight": "[75-100)",
    "admission_type_id": 1,
    "admission_type_desc": ADMISSION_TYPE_LOOKUP[1],
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "payer_code": "MC",
    "medical_specialty": "Cardiology",
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 12,
    "number_diagnoses": 5,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1,
    "A1Cresult": "Norm",
    "max_glu_serum": "None",
    "diag_2": "401",
    "diag_3": "250",
    "diabetesMed": "Yes",
    "metformin": "Steady",
    "insulin": "Steady",
}

for med_field in MEDICATION_FIELDS:
    DEFAULT_PATIENT.setdefault(med_field, "No")

SAMPLE_PATIENTS: Dict[str, Dict[str, object]] = {
    "Post-discharge follow-up": {
        **DEFAULT_PATIENT,
        "gender": "Male",
        "age": "[50-60)",
        "admission_type_id": 2,
        "admission_type_desc": ADMISSION_TYPE_LOOKUP[2],
        "discharge_disposition_id": 5,
        "admission_source_id": 2,
        "num_lab_procedures": 52,
        "num_medications": 16,
        "number_inpatient": 2,
        "number_diagnoses": 7,
        "A1Cresult": ">7",
        "max_glu_serum": ">200",
        "diag_2": "414",
        "diag_3": "786",
        "metformin": "Up",
        "insulin": "Steady",
        "pioglitazone": "Steady",
    },
    "High-utilization patient": {
        **DEFAULT_PATIENT,
        "age": "[70-80)",
        "weight": "[100-125)",
        "admission_type_id": 1,
        "admission_type_desc": ADMISSION_TYPE_LOOKUP[1],
        "discharge_disposition_id": 3,
        "admission_source_id": 7,
        "num_lab_procedures": 60,
        "num_medications": 20,
        "number_outpatient": 3,
        "number_emergency": 1,
        "number_inpatient": 4,
        "number_diagnoses": 9,
        "A1Cresult": ">8",
        "max_glu_serum": ">300",
        "diag_2": "428",
        "diag_3": "250",
        "insulin": "Up",
        "rosiglitazone": "Steady",
        "pioglitazone": "Steady",
    },
}

if "patient_defaults" not in st.session_state:
    st.session_state.patient_defaults = DEFAULT_PATIENT.copy()

st.title("🏥 Federated Health App")
st.caption("Predict readmission risk, likely length of stay, medication changes, and diagnosis grouping from structured encounter data.")

with st.sidebar:
    st.header("Configuration")
    st.markdown("Adjust model decision thresholds and load ready-made patient profiles to explore the predictions.")
    sample_name = st.selectbox("Patient examples", list(SAMPLE_PATIENTS.keys()), index=0)
    if st.button("Load selected example"):
        st.session_state.patient_defaults = SAMPLE_PATIENTS[sample_name].copy()
        st.experimental_rerun()
    st.divider()
    st.subheader("Decision thresholds")
    readm_thr = st.slider(
        "Readmission alert threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(DEFAULT_THRESHOLDS["readmission"]),
        step=0.01,
    )
    med_change_thr = st.slider(
        "Medication change alert threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(DEFAULT_THRESHOLDS["medication_change"]),
        step=0.01,
    )
    st.caption("Risk levels are categorized as Low (<34%), Moderate (34-67%), or High (>67%).")

st.markdown("""
Provide as much detail as possible about the encounter. Completing optional context such as discharge disposition or payer information helps the model avoid falling back to generic defaults, improving predictive accuracy.
""")

form_defaults = st.session_state.patient_defaults

with st.form("patient_form"):
    st.subheader("Patient profile")
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    with demo_col1:
        race = st.selectbox("Race", RACE_OPTIONS, index=RACE_OPTIONS.index(form_defaults.get("race", RACE_OPTIONS[0])))
        gender = st.selectbox("Gender", GENDER_OPTIONS, index=GENDER_OPTIONS.index(form_defaults.get("gender", GENDER_OPTIONS[0])))
        age = st.selectbox("Age band", AGE_OPTIONS, index=AGE_OPTIONS.index(form_defaults.get("age", AGE_OPTIONS[0])))
    with demo_col2:
        weight = st.selectbox("Weight band", WEIGHT_OPTIONS, index=WEIGHT_OPTIONS.index(form_defaults.get("weight", WEIGHT_OPTIONS[0])), help="Weight from most recent inpatient stay (if coded).")
        payer_code = st.selectbox("Primary payer", PAYER_OPTIONS, index=PAYER_OPTIONS.index(form_defaults.get("payer_code", PAYER_OPTIONS[0])))
        medical_specialty = st.selectbox("Admitting specialty", MEDICAL_SPECIALTY_OPTIONS, index=MEDICAL_SPECIALTY_OPTIONS.index(form_defaults.get("medical_specialty", MEDICAL_SPECIALTY_OPTIONS[0])))
    with demo_col3:
        admission_type_id = st.selectbox(
            "Admission type",
            options=list(ADMISSION_TYPE_LOOKUP.keys()),
            format_func=lambda x: f"{x} – {ADMISSION_TYPE_LOOKUP[x]}",
            index=list(ADMISSION_TYPE_LOOKUP.keys()).index(int(form_defaults.get("admission_type_id", 1))),
        )
        discharge_disposition_id = st.selectbox(
            "Discharge disposition",
            options=list(DISCHARGE_OPTIONS.keys()),
            format_func=lambda x: f"{x} – {DISCHARGE_OPTIONS[x]}",
            index=list(DISCHARGE_OPTIONS.keys()).index(int(form_defaults.get("discharge_disposition_id", 1))),
        )
        admission_source_id = st.selectbox(
            "Admission source",
            options=list(ADMISSION_SOURCE_OPTIONS.keys()),
            format_func=lambda x: f"{x} – {ADMISSION_SOURCE_OPTIONS[x]}",
            index=list(ADMISSION_SOURCE_OPTIONS.keys()).index(int(form_defaults.get("admission_source_id", 7))),
        )

    st.markdown("---")
    st.subheader("Utilization & labs")
    util_col1, util_col2, util_col3 = st.columns(3)
    with util_col1:
        num_lab_procedures = st.number_input("Lab procedures", min_value=0, max_value=120, value=int(form_defaults.get("num_lab_procedures", 40)))
        num_procedures = st.number_input("Procedures", min_value=0, max_value=10, value=int(form_defaults.get("num_procedures", 1)))
        num_medications = st.number_input("Medications", min_value=0, max_value=60, value=int(form_defaults.get("num_medications", 10)))
    with util_col2:
        number_outpatient = st.number_input("Outpatient visits (past year)", min_value=0, max_value=50, value=int(form_defaults.get("number_outpatient", 0)))
        number_emergency = st.number_input("Emergency visits (past year)", min_value=0, max_value=50, value=int(form_defaults.get("number_emergency", 0)))
        number_inpatient = st.number_input("Inpatient stays (past year)", min_value=0, max_value=50, value=int(form_defaults.get("number_inpatient", 1)))
    with util_col3:
        number_diagnoses = st.number_input("Unique diagnoses this stay", min_value=0, max_value=20, value=int(form_defaults.get("number_diagnoses", 5)))
        a1c = st.selectbox("A1C result", A1C_OPTIONS, index=A1C_OPTIONS.index(form_defaults.get("A1Cresult", A1C_OPTIONS[0])), help="Most recent A1C in the medical record.")
        glu = st.selectbox("Max glucose serum", GLUCOSE_OPTIONS, index=GLUCOSE_OPTIONS.index(form_defaults.get("max_glu_serum", GLUCOSE_OPTIONS[0])), help="Highest glucose serum measurement recorded.")

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        diag_2 = st.text_input("Secondary diagnosis code", value=str(form_defaults.get("diag_2", "UNK")), help="Use ICD-9 style codes (e.g., 401 for hypertension).").strip().upper()
    with diag_col2:
        diag_3 = st.text_input("Tertiary diagnosis code", value=str(form_defaults.get("diag_3", "UNK")), help="Use ICD-9 style codes (e.g., 250 for diabetes).").strip().upper()

    med_values: Dict[str, str] = {}
    with st.expander("Medication orders", expanded=False):
        med_columns = st.columns(3)
        meds_collections = [
            MEDICATION_FIELDS[i::3] for i in range(3)
        ]
        for column, fields in zip(med_columns, meds_collections):
            with column:
                for med_field in fields:
                    default_val = form_defaults.get(med_field, "No")
                    med_values[med_field] = st.selectbox(
                        med_field.replace("-", " → ").title(),
                        MEDICATION_CATEGORIES,
                        index=MEDICATION_CATEGORIES.index(default_val if default_val in MEDICATION_CATEGORIES else "No"),
                        help="Order status during this admission.",
                        key=f"med_{med_field}",
                    )

    diabetes_med = st.selectbox(
        "Patient on any diabetes medications?",
        ["Yes", "No"],
        index=["Yes", "No"].index(form_defaults.get("diabetesMed", "Yes")),
    )

    submitted = st.form_submit_button("Run prediction", width="stretch")

patient_inputs = {
    "race": race,
    "gender": gender,
    "age": age,
    "weight": weight,
    "payer_code": payer_code,
    "medical_specialty": medical_specialty,
    "admission_type_id": admission_type_id,
    "admission_type_desc": ADMISSION_TYPE_LOOKUP.get(admission_type_id),
    "discharge_disposition_id": discharge_disposition_id,
    "admission_source_id": admission_source_id,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_diagnoses": number_diagnoses,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "A1Cresult": a1c,
    "max_glu_serum": glu,
    "diag_2": diag_2,
    "diag_3": diag_3,
    "diabetesMed": diabetes_med,
}
patient_inputs.update(med_values)

if submitted:
    st.session_state.patient_defaults = patient_inputs.copy()
    thresholds = {
        "readmission": readm_thr,
        "medication_change": med_change_thr,
    }
    with st.spinner("Scoring patient across all tasks..."):
        results = predict_patient(
            patient_inputs,
            thresholds=thresholds,
            include_distributions=True,
            top_k=3,
        )

    st.subheader("Prediction summary")

    def render_binary_card(title: str, info: Dict[str, object]):
        prob = float(info.get("Probability", 0.0))
        probability_pct = f"{prob:.1f}%"
        risk = info.get("Risk", "Unknown")
        risk_emoji = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}.get(risk, "⚪")
        threshold = float(info.get("Threshold", 0.5))
        st.markdown(f"#### {title}")
        st.metric("Prediction", info.get("Prediction", "N/A"), probability_pct)
        st.progress(min(max(prob / 100.0, 0.0), 1.0))
        st.caption(f"{risk_emoji} Risk level: **{risk}** • Alert threshold: {threshold:.2f}")

    col_left, col_right = st.columns(2)
    with col_left:
        render_binary_card("Readmission within 30 days", results["Readmission (<30d)"])
        render_binary_card("Medication regimen change", results["Medication Change"])
    with col_right:
        los_info = results["Length of Stay"]
        st.markdown("#### Length of stay")
        st.metric("Most likely band", los_info.get("Prediction", "N/A"), f"{los_info.get('Probability', 0):.1f}% confidence")
        st.progress(min(max(float(los_info.get("Probability", 0)) / 100.0, 0.0), 1.0))
        top_los_df = pd.DataFrame(los_info.get("Top Alternatives", [])).rename(columns={"Label": "Length of stay", "Probability": "Probability (%)"})
        if not top_los_df.empty:
            st.dataframe(top_los_df, hide_index=True, width="stretch")

        diag_info = results["Diagnosis Group"]
        st.markdown("#### Diagnosis grouping")
        st.metric("Most likely group", diag_info.get("Prediction", "N/A"), f"{diag_info.get('Probability', 0):.1f}% confidence")
        diag_df = pd.DataFrame(diag_info.get("Top Alternatives", [])).rename(columns={"Label": "Group", "Probability": "Probability (%)"})
        if not diag_df.empty:
            st.dataframe(diag_df, hide_index=True, width="stretch")

    st.markdown("---")
    st.subheader("Detailed outputs")
    tabs = st.tabs(["Distributions", "Raw JSON"])
    with tabs[0]:
        distributions = results.get("_distributions", {})
        if distributions:
            los_dist = distributions.get("length_of_stay", {})
            if los_dist:
                st.markdown("**Length of stay probabilities**")
                st.bar_chart(pd.Series(los_dist))
            diag_dist = distributions.get("diagnosis_group", {})
            if diag_dist:
                st.markdown("**Diagnosis group probabilities**")
                st.bar_chart(pd.Series(diag_dist))
        else:
            st.info("Distribution details not available for this model.")
    with tabs[1]:
        st.code(json.dumps(results, indent=2), language="json")

    st.markdown("---")
    st.subheader("Visual analytics")

    readm_prob_pct = float(results["Readmission (<30d)"]["Probability"])
    los_dist = distributions.get("length_of_stay", {}) if distributions else {}
    diag_dist = distributions.get("diagnosis_group", {}) if distributions else {}

    chart_row_one = st.columns(2)

    if los_dist:
        los_labels = list(los_dist.keys())
        los_values = [float(prob) * 100 for prob in los_dist.values()]
        fig_los, ax_los = plt.subplots(figsize=(6, 4))
        ax_los.bar(los_labels, los_values, color=["#4E79A7", "#F28E2B", "#76B7B2"][: len(los_labels)])
        ax_los.set_ylabel("Probability (%)")
        ax_los.set_title("Length of stay distribution")
        for idx, value in enumerate(los_values):
            ax_los.text(idx, value + 1, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        with chart_row_one[0]:
            st.pyplot(fig_los)
        plt.close(fig_los)
    else:
        with chart_row_one[0]:
            st.info("Length of stay distribution not available.")

    pie_values = [readm_prob_pct, max(0.0, 100 - readm_prob_pct)]
    pie_labels = ["Predicted readmission risk", "Remaining margin"]
    fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
    colors = ["#E15759", "#9D9D9D"]
    wedges, texts, autotexts = ax_pie.pie(
        pie_values,
        labels=pie_labels,
        autopct="%1.1f%%",
        startangle=120,
        colors=colors,
        textprops={"color": "white"},
    )
    for text in texts:
        text.set_color("black")
    ax_pie.set_title("Readmission probability breakdown")
    with chart_row_one[1]:
        st.pyplot(fig_pie)
    plt.close(fig_pie)

    chart_row_two = st.columns(2)

    if diag_dist:
        diag_probs = [float(prob) * 100 for prob in diag_dist.values()]
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(diag_probs, bins=min(10, max(4, len(diag_probs))), color="#59A14F", edgecolor="#F0F0F0")
        ax_hist.set_xlabel("Diagnosis probability (%)")
        ax_hist.set_ylabel("Number of groups")
        ax_hist.set_title("Diagnosis confidence spread")
        plt.tight_layout()
        with chart_row_two[0]:
            st.pyplot(fig_hist)
        plt.close(fig_hist)
    else:
        with chart_row_two[0]:
            st.info("Diagnosis probability histogram not available.")

    threshold_grid = np.linspace(5, 95, 19)
    margin = readm_prob_pct - threshold_grid
    fig_line, ax_line = plt.subplots(figsize=(6, 4))
    ax_line.plot(threshold_grid, margin, color="#4E79A7", marker="o", linewidth=2)
    ax_line.axhline(0, color="#9D9D9D", linestyle="--", linewidth=1)
    ax_line.axvline(readm_thr * 100, color="#E15759", linestyle=":", linewidth=1)
    ax_line.scatter([readm_thr * 100], [readm_prob_pct - readm_thr * 100], color="#E15759", zorder=3)
    ax_line.set_xlabel("Alert threshold (%)")
    ax_line.set_ylabel("Probability - threshold (pp)")
    ax_line.set_title("Readmission margin across thresholds")
    ax_line.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    with chart_row_two[1]:
        st.pyplot(fig_line)
    plt.close(fig_line)

    med_status_to_score = {"Down": -1, "No": 0, "Steady": 1, "Up": 2}
    med_chart_df = pd.DataFrame(
        [
            {
                "Medication": med.replace("-", " → ").title(),
                "Status": status,
                "Score": med_status_to_score.get(status, 0),
            }
            for med, status in med_values.items()
        ]
    )

    if not med_chart_df.empty:
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 4))
        color_map = {"Down": "#E15759", "No": "#9D9D9D", "Steady": "#59A14F", "Up": "#EDC948"}
        x_positions = np.arange(len(med_chart_df))
        scatter_colors = [color_map.get(status, "#4E79A7") for status in med_chart_df["Status"]]
        ax_scatter.scatter(x_positions, med_chart_df["Score"], s=140, c=scatter_colors, edgecolors="white", linewidths=0.8)
        ax_scatter.set_xticks(x_positions)
        ax_scatter.set_xticklabels(med_chart_df["Medication"], rotation=45, ha="right")
        ax_scatter.set_yticks([-1, 0, 1, 2])
        ax_scatter.set_yticklabels(["Down", "No", "Steady", "Up"])
        ax_scatter.set_ylabel("Order intensity")
        ax_scatter.set_title("Medication order directions")
        ax_scatter.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)
    else:
        st.info("No medication orders supplied to visualize.")
