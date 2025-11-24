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

## Improving model accuracy

The repository now includes a LightGBM training pipeline that mirrors the
features collected in the Streamlit app.  To refresh the model with your own
data (and automatically compute better decision thresholds):

1. Prepare a CSV file containing the feature columns used by the app together
   with the target labels (`readmitted_30d`, `length_of_stay`,
   `medication_change`, and `diagnosis_group`).
2. Install the dependencies listed in `requirements.txt` (the pinned
   `scikit-learn` version matches the version used during training to avoid
   compatibility drift).
3. Run the trainer:

   ```bash
   python training/train_pipeline.py --input path/to/your_dataset.csv
   ```

   The script performs a stratified train/validation split, fits the
   multi-output LightGBM model, evaluates hold-out metrics, searches for
   F1-optimal thresholds for the binary tasks, and persists the refreshed
   artefacts in `models/`.
4. Restart the Streamlit dashboard; the app automatically picks up the updated
   model, feature medians, and recommended thresholds for the readmission and
   medication change alerts.
