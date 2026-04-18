from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
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


class Predictor:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        artifacts_dir = repo_root / "ml" / "venv" / "models" / "artifacts"

        self.cancer = self._load(
            checkpoint_path=artifacts_dir / "cancer_model.pt",
            preprocessor_path=artifacts_dir / "cancer_preprocessor.pkl",
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
            positive_label="malignant",
            negative_label="benign",
        )
        self.blood = self._load(
            checkpoint_path=artifacts_dir / "diabetes_model.pt",
            preprocessor_path=artifacts_dir / "diabetes_preprocessor.pkl",
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
            positive_label="high_diabetes_risk",
            negative_label="low_diabetes_risk",
        )
        self.cardio = self._load(
            checkpoint_path=artifacts_dir / "framingham_model.pt",
            preprocessor_path=artifacts_dir / "framingham_preprocessor.pkl",
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
            positive_label="high_chd_risk",
            negative_label="low_chd_risk",
        )

    def _load(
        self,
        checkpoint_path: Path,
        preprocessor_path: Path,
        features: list[str],
        positive_label: str,
        negative_label: str,
    ) -> LoadedModel:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = TabularBinaryClassifier(
            input_dim=int(checkpoint["input_dim"]),
            hidden1=int(checkpoint["hidden1"]),
            hidden2=int(checkpoint["hidden2"]),
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return LoadedModel(
            model=model,
            preprocessor=joblib.load(preprocessor_path),
            threshold=float(checkpoint.get("threshold", 0.5)),
            positive_label=positive_label,
            negative_label=negative_label,
            features=features,
        )

    @staticmethod
    def _predict(loaded: LoadedModel, payload: dict) -> tuple[str, float]:
        frame = pd.DataFrame([payload], columns=loaded.features)
        transformed = loaded.preprocessor.transform(frame)
        tensor = torch.tensor(transformed, dtype=torch.float32)
        with torch.no_grad():
            probability = torch.sigmoid(loaded.model(tensor)).item()
        label = loaded.positive_label if probability >= loaded.threshold else loaded.negative_label
        return label, probability

    def predict_cancer(self, payload: dict) -> tuple[str, float]:
        return self._predict(self.cancer, payload)

    def predict_blood(self, payload: dict) -> tuple[str, float]:
        return self._predict(self.blood, payload)

    def predict_cardio(self, payload: dict) -> tuple[str, float]:
        return self._predict(self.cardio, payload)
