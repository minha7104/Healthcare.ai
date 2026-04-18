# Healthcare AI: Multi-Disease Risk Prediction System

Production-oriented healthcare ML system with three tabular prediction models and a FastAPI backend:

- Cardiovascular risk (`framingham.csv`, target `TenYearCHD`)
- Breast cancer classification (`cancer.csv`, target `diagnosis`)
- Diabetes risk (`diabetes.csv`, target `Outcome`)

## Problem Statement

Clinical risk-screening models must balance two objectives:

- strong ranking ability (ROC-AUC),
- strong recall for high-risk patients.

This project implements an end-to-end pipeline from training to deployable API, with model evaluation, baseline comparison, and explainability.

## Project Structure

```text
backend/
├── models/              # .pt and metrics json copied for serving
├── preprocessors/       # .pkl preprocessors for serving
├── routes/
│   ├── cancer.py
│   ├── diabetes.py
│   └── cardio.py
├── utils/
│   └── predictor_service.py
├── main.py
└── requirements.txt

ml/venv/
├── training/            # train_framingham.py, train_cancer.py, train_diabetes.py
├── models/artifacts/    # trained artifacts + metrics + evaluation plots
└── utils/               # preprocessing utilities
```

## ML Pipeline (All Datasets)

1. Load CSV dataset
2. Identify target column
3. Stratified train-test split
4. Missing value handling with `SimpleImputer`
5. `ColumnTransformer`:
   - `StandardScaler` for continuous features
   - binary features kept unscaled
6. Convert processed arrays to PyTorch tensors
7. Neural architecture: `Input -> 32/64/... -> 16/32/... -> 1` with ReLU + Dropout
8. `BCEWithLogitsLoss`
9. Class imbalance handling via `pos_weight`
10. Mini-batch PyTorch training loop (`DataLoader`)

## Baseline and Tuning

Each task compares tuned neural network variants against a `LogisticRegression(class_weight="balanced")` baseline.

Tuning dimensions used:

- learning rate
- hidden-layer sizes
- dropout
- batch size
- `pos_weight` scaling
- decision threshold for recall-sensitive operation

## Evaluation Results

### Framingham (Cardio)

- **Neural Net**: ROC-AUC `0.6794`, Precision `0.1902`, Recall `0.8760`, F1 `0.3126`
- **Logistic Baseline**: ROC-AUC `0.7008`, Precision `0.1687`, Recall `0.9535`, F1 `0.2867`
- **Conclusion**: Model needs improvement (AUC below target)

### Cancer

- **Neural Net**: ROC-AUC `0.9940`, Precision `1.0000`, Recall `0.9286`, F1 `0.9630`
- **Logistic Baseline**: ROC-AUC `0.9954`, Precision `0.9756`, Recall `0.9524`, F1 `0.9639`
- **Conclusion**: Model is acceptable

### Diabetes

- **Neural Net**: ROC-AUC `0.8109`, Precision `0.5263`, Recall `0.9259`, F1 `0.6711`
- **Logistic Baseline**: ROC-AUC `0.8248`, Precision `0.5253`, Recall `0.9630`, F1 `0.6797`
- **Conclusion**: Model is acceptable

## Explainability

Explainability is provided via permutation importance (ROC-AUC scoring) generated during training and stored in:

- `ml/venv/models/artifacts/*_metrics.json`

For serving-time responses, API returns `top_features` by combining:

- global permutation importance ranking
- per-request transformed feature magnitude

Example interpretation:

- High diabetes risk due to: `Glucose`, `BMI`, `Age`

## Evaluation Artifacts

Saved in `ml/venv/models/artifacts/evaluation/`:

- `*_confusion_matrix.png`
- `*_roc_curve.png`

## API Endpoints

- `POST /predict/cancer`
- `POST /predict/blood`
- `POST /predict/cardio`
- `GET /health`

Response format:

```json
{
  "prediction": "High Risk",
  "probability": 0.73,
  "top_features": ["glucose", "BMI", "age"]
}
```

## Deployment (Render Ready)

From `backend/`:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 10000
```

Render start command:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

## Notes for Reviewers

- Training and serving are fully separated.
- Models and preprocessors are loaded at startup, not per-request.
- Metrics include neural vs baseline comparison and explicit acceptability conclusions.
