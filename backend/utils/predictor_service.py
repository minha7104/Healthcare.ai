from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model_definition import TabularBinaryClassifier


@dataclass
class LoadedModel:
    model: TabularBinaryClassifier
    preprocessor: object
    threshold: float
    positive_label: str
    negative_label: str
    features: list[str]
    global_feature_importance: dict[str, float]


class PredictorService:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source_artifacts_dir = root / "ml" / "venv" / "models" / "artifacts"
        model_dir = root / "backend" / "models"
        preprocessor_dir = root / "backend" / "preprocessors"
        model_dir.mkdir(parents=True, exist_ok=True)
        preprocessor_dir.mkdir(parents=True, exist_ok=True)

        self.cancer = self._load_bundle(
            source_artifacts_dir=source_artifacts_dir,
            model_dir=model_dir,
            preprocessor_dir=preprocessor_dir,
            model_name="cancer",
            features=[
                "radius_mean",
                "texture_mean",
                "perimeter_mean",
                "area_mean",
                "smoothness_mean",
                "compactness_mean",
                "concavity_mean",
                "concave points_mean",
                "symmetry_mean",
                "fractal_dimension_mean",
                "radius_se",
                "texture_se",
                "perimeter_se",
                "area_se",
                "smoothness_se",
                "compactness_se",
                "concavity_se",
                "concave points_se",
                "symmetry_se",
                "fractal_dimension_se",
                "radius_worst",
                "texture_worst",
                "perimeter_worst",
                "area_worst",
                "smoothness_worst",
                "compactness_worst",
                "concavity_worst",
                "concave points_worst",
                "symmetry_worst",
                "fractal_dimension_worst",
            ],
            positive_label="Malignant",
            negative_label="Benign",
        )
        self.diabetes = self._load_bundle(
            source_artifacts_dir=source_artifacts_dir,
            model_dir=model_dir,
            preprocessor_dir=preprocessor_dir,
            model_name="diabetes",
            features=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ],
            positive_label="High Risk",
            negative_label="Low Risk",
        )
        self.cardio = self._load_bundle(
            source_artifacts_dir=source_artifacts_dir,
            model_dir=model_dir,
            preprocessor_dir=preprocessor_dir,
            model_name="framingham",
            features=[
                "male",
                "age",
                "education",
                "currentSmoker",
                "cigsPerDay",
                "BPMeds",
                "prevalentStroke",
                "prevalentHyp",
                "diabetes",
                "totChol",
                "sysBP",
                "diaBP",
                "BMI",
                "heartRate",
                "glucose",
            ],
            positive_label="High Risk",
            negative_label="Low Risk",
        )

    def _load_bundle(
        self,
        source_artifacts_dir: Path,
        model_dir: Path,
        preprocessor_dir: Path,
        model_name: str,
        features: list[str],
        positive_label: str,
        negative_label: str,
    ) -> LoadedModel:
        source_model = source_artifacts_dir / f"{model_name}_model.pt"
        source_preprocessor = source_artifacts_dir / f"{model_name}_preprocessor.pkl"
        source_metrics = source_artifacts_dir / f"{model_name}_metrics.json"
        target_model = model_dir / f"{model_name}_model.pt"
        target_preprocessor = preprocessor_dir / f"{model_name}_preprocessor.pkl"
        target_metrics = model_dir / f"{model_name}_metrics.json"

        target_model.write_bytes(source_model.read_bytes())
        target_preprocessor.write_bytes(source_preprocessor.read_bytes())
        target_metrics.write_text(source_metrics.read_text(encoding="utf-8"), encoding="utf-8")

        checkpoint = torch.load(target_model, map_location="cpu")
        model = TabularBinaryClassifier(
            input_dim=int(checkpoint["input_dim"]),
            hidden1=int(checkpoint["hidden1"]),
            hidden2=int(checkpoint["hidden2"]),
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        metrics_payload = json.loads(target_metrics.read_text(encoding="utf-8"))
        importance_entries = metrics_payload.get("feature_importance", [])
        global_importance = {
            entry["feature"]: max(float(entry["importance_mean"]), 0.0) for entry in importance_entries
        }

        return LoadedModel(
            model=model,
            preprocessor=joblib.load(target_preprocessor),
            threshold=float(checkpoint.get("threshold", 0.5)),
            positive_label=positive_label,
            negative_label=negative_label,
            features=features,
            global_feature_importance=global_importance,
        )

    @staticmethod
    def _top_features(loaded: LoadedModel, payload: dict, transformed_row: np.ndarray, top_k: int = 3) -> list[str]:
        row_scores = []
        for idx, feature in enumerate(loaded.features):
            global_score = loaded.global_feature_importance.get(feature, 0.0)
            local_magnitude = abs(float(transformed_row[idx]))
            row_scores.append((feature, global_score * (1.0 + local_magnitude)))
        row_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in row_scores[:top_k]]

    @staticmethod
    def _predict(loaded: LoadedModel, payload: dict) -> tuple[str, float, list[str]]:
        frame = pd.DataFrame([payload], columns=loaded.features)
        transformed = loaded.preprocessor.transform(frame)
        transformed_row = np.asarray(transformed).reshape(-1)
        tensor = torch.tensor(transformed, dtype=torch.float32)
        with torch.no_grad():
            probability = torch.sigmoid(loaded.model(tensor)).item()
        label = loaded.positive_label if probability >= loaded.threshold else loaded.negative_label
        top_features = PredictorService._top_features(loaded, payload, transformed_row)
        return label, probability, top_features

    def predict_cancer(self, payload: dict) -> tuple[str, float, list[str]]:
        return self._predict(self.cancer, payload)

    def predict_blood(self, payload: dict) -> tuple[str, float, list[str]]:
        return self._predict(self.diabetes, payload)

    def predict_cardio(self, payload: dict) -> tuple[str, float, list[str]]:
        return self._predict(self.cardio, payload)
