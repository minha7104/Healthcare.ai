# Healthcare AI Backend API

FastAPI service for three healthcare prediction models:

- `/predict/cancer` (Breast Cancer Wisconsin)
- `/predict/blood` (Diabetes risk)
- `/predict/cardio` (10-year CHD risk)

## Setup

From the `backend` directory:

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

Health check:

- `GET /health`

## Response Format

All prediction endpoints return:

```json
{
  "prediction": "string_label",
  "probability": 0.0,
  "top_features": ["feature_1", "feature_2", "feature_3"]
}
```

## Endpoint Examples

### 1) Cancer

`POST /predict/cancer`

```json
{
  "radius_mean": 14.0,
  "texture_mean": 20.0,
  "perimeter_mean": 90.0,
  "area_mean": 600.0,
  "smoothness_mean": 0.1,
  "compactness_mean": 0.1,
  "concavity_mean": 0.08,
  "concave points_mean": 0.05,
  "symmetry_mean": 0.18,
  "fractal_dimension_mean": 0.06,
  "radius_se": 0.4,
  "texture_se": 1.2,
  "perimeter_se": 2.5,
  "area_se": 30.0,
  "smoothness_se": 0.006,
  "compactness_se": 0.02,
  "concavity_se": 0.03,
  "concave points_se": 0.01,
  "symmetry_se": 0.02,
  "fractal_dimension_se": 0.003,
  "radius_worst": 16.0,
  "texture_worst": 25.0,
  "perimeter_worst": 100.0,
  "area_worst": 700.0,
  "smoothness_worst": 0.14,
  "compactness_worst": 0.2,
  "concavity_worst": 0.2,
  "concave points_worst": 0.1,
  "symmetry_worst": 0.3,
  "fractal_dimension_worst": 0.08
}
```

### 2) Diabetes (Blood)

`POST /predict/blood`

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 28.1,
  "DiabetesPedigreeFunction": 0.3,
  "Age": 33
}
```

### 3) Cardio (Framingham CHD)

`POST /predict/cardio`

```json
{
  "male": 1,
  "age": 55,
  "education": 2,
  "currentSmoker": 1,
  "cigsPerDay": 15,
  "BPMeds": 0,
  "prevalentStroke": 0,
  "prevalentHyp": 1,
  "diabetes": 0,
  "totChol": 230,
  "sysBP": 140,
  "diaBP": 90,
  "BMI": 28,
  "heartRate": 75,
  "glucose": 95
}
```

## Notes

- The API applies saved preprocessing pipelines automatically.
- Predictions are generated using saved PyTorch model checkpoints.
- `probability` is the sigmoid output; endpoint-specific thresholds decide the class label.
- `top_features` provides explainability signals using global permutation importance combined with per-request feature magnitude.
