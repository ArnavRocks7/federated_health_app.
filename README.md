# Federated Health App

A simple Streamlit dashboard that loads a trained multi-output healthcare model.

## Features
- Predicts readmission (<30d)
- Predicts Length of Stay (short/medium/long)
- Predicts medication change
- Predicts diagnosis group

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```
