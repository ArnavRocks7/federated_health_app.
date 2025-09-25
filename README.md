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
3. (Optional) Run a quick hyperparameter search to squeeze extra performance:

   ```bash
   python training/train_pipeline.py \
       --input path/to/your_dataset.csv \
       --search --search-iterations 15
   ```

   Add `--no-refit` if you would like to inspect the validation metrics before
   refitting on the full dataset. When the search flag is omitted, the pipeline
   trains with well-performing default parameters.

4. Run the trainer:

   ```bash
   python training/train_pipeline.py --input path/to/your_dataset.csv
   ```

   The script performs a stratified train/validation split, fits the
   multi-output LightGBM model, evaluates hold-out metrics, searches for
   F1-optimal thresholds for the binary tasks, and persists the refreshed
   artefacts in `models/`.
5. Restart the Streamlit dashboard; the app automatically picks up the updated
   model, feature medians, and recommended thresholds for the readmission and
   medication change alerts.

   > **Note:** If you encounter an `InconsistentVersionWarning` from scikit-learn
   > while launching the app, reinstall the pinned dependency to align with the
   > trained artefacts:
   >
   > ```bash
   > pip install --upgrade --force-reinstall scikit-learn==1.6.1
   > ```

### Strategies to boost predictive accuracy

Before rerunning the trainer, consider the following workflow to systematically
lift model performance:

- **Audit data quality** – remove duplicate encounters, fill implausible values,
  and ensure ICD codes are harmonised to the groupings expected by the model.
- **Balance the labels** – the readmission and medication-change tasks are
  typically imbalanced; upsample minority cases or apply class weights to avoid
  models that always predict the majority class.
- **Engineer richer features** – join longitudinal history (e.g. number of
  prior admissions, comorbidity indices) and normalise continuous labs to
  highlight meaningful variation.
- **Tune hyperparameters** – use the built-in `--search` flag of
  `train_pipeline.py` (see `python training/train_pipeline.py --help`) or a
  library such as Optuna to explore learning rate, tree depth, and regularisers.
- **Calibrate probabilities** – evaluate Brier score/expected calibration error
  and fit isotonic or Platt scaling on the validation split when necessary.
- **Track metrics beyond accuracy** – optimise for F1, recall, or precision at
  clinically relevant thresholds so the model aligns with downstream decision
  needs.
- **Align runtime dependencies** – deploy with the same `scikit-learn` version
  the artefacts were trained with (1.6.1) to prevent unpickling issues and
  unexpected scoring differences.
