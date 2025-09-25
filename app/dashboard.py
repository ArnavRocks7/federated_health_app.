import streamlit as st, json
from inference import predict_patient

st.set_page_config(page_title="Healthcare Dashboard", layout="wide")
st.title("🏥 Federated Health App")
st.caption("Multi-output predictions: readmission, LOS, med change, diagnosis group")

with st.form("patient_form"):
    st.subheader("Patient Input")
    col1, col2, col3 = st.columns(3)
    with col1:
        race  = st.selectbox("Race", ["Caucasian","AfricanAmerican","Asian","Hispanic","Other"])
        gender= st.selectbox("Gender", ["Male","Female"])
        age   = st.selectbox("Age Range", ["[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)"])
        medical_specialty = st.text_input("Medical Specialty", value="Cardiology")
    with col2:
        num_lab_procedures = st.number_input("Lab procedures", 0, 100, 45)
        num_procedures     = st.number_input("Procedures", 0, 10, 1)
        num_medications    = st.number_input("Medications", 0, 50, 12)
        number_diagnoses   = st.number_input("Number of diagnoses", 0, 20, 5)
    with col3:
        number_outpatient  = st.number_input("Outpatient visits", 0, 30, 0)
        number_emergency   = st.number_input("Emergency visits", 0, 30, 0)
        number_inpatient   = st.number_input("Inpatient visits", 0, 30, 1)
        admission_type_id  = st.number_input("Admission type id", 1, 8, 1)
    a1c  = st.selectbox("A1C result", ["Norm",">7",">8","None"])
    glu  = st.selectbox("Max glucose serum", ["None",">200",">300","Norm"])
    diag_2 = st.text_input("Diagnosis code 2", value="401")
    diag_3 = st.text_input("Diagnosis code 3", value="250")
    metformin = st.selectbox("Metformin", ["No","Steady","Up","Down"])
    insulin   = st.selectbox("Insulin",   ["No","Steady","Up","Down"])
    diabetesMed = st.selectbox("On diabetes medication?", ["Yes","No"])
    submitted = st.form_submit_button("Predict")

if submitted:
    patient = {
        "race": race, "gender": gender, "age": age, "medical_specialty": medical_specialty,
        "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
        "num_medications": num_medications, "number_diagnoses": number_diagnoses,
        "number_outpatient": number_outpatient, "number_emergency": number_emergency,
        "number_inpatient": number_inpatient, "admission_type_id": admission_type_id,
        "A1Cresult": a1c, "max_glu_serum": glu, "diag_2": diag_2, "diag_3": diag_3,
        "metformin": metformin, "insulin": insulin, "diabetesMed": diabetesMed,
    }
    results = predict_patient(patient, thr=0.5)
    st.subheader("Results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Readmission (<30d)", results["Readmission (<30d)"]["Prediction"],
                  f'{results["Readmission (<30d)"]["Probability"]}%')
        st.metric("Medication Change", results["Medication Change"]["Prediction"],
                  f'{results["Medication Change"]["Probability"]}%')
    with c2:
        st.metric("Length of Stay", results["Length of Stay"]["Prediction"],
                  f'{results["Length of Stay"]["Probability"]}%')
        st.metric("Diagnosis Group", results["Diagnosis Group"]["Prediction"],
                  f'{results["Diagnosis Group"]["Probability"]}%')
    st.code(json.dumps(results, indent=2), language="json")
