from pathlib import Path
from typing import Dict, Any, Optional

from dataclasses import dataclass
import joblib
import pandas as pd

from src.features import add_features

from src.config import config

DEFAULT_MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / config.name
    / config.version
    / "model.pkl"
)


@dataclass
class PredictionResult:
    prediction: int
    label: str
    probability: Optional[float]


class FraudModel:

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):

        if not model_path.exists():

            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.expected_features = self.model.named_steps["preprocess"].get_feature_names_out()

    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:

        required_fields = [
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]

        missing = [f for f in required_fields if f not in data]

        if missing:
            
            raise ValueError(f"Missing required fields: {missing}")

        try:
            df = pd.DataFrame([{
                "type": data["type"],
                "amount": float(data["amount"]),
                "oldbalanceOrg": float(data["oldbalanceOrg"]),
                "newbalanceOrig": float(data["newbalanceOrig"]),
                "oldbalanceDest": float(data["oldbalanceDest"]),
                "newbalanceDest": float(data["newbalanceDest"]),
            }])

        except (TypeError, ValueError) as e:

            raise ValueError(f"Invalid input types: {e}")
        
        if df.isnull().any().any():
            raise ValueError("Dataset содержит NaN до feature engineering")

        df = add_features(df)

        return df

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:

        df = self._prepare_dataframe(data)

        prediction = int(self.model.predict(df)[0])

        probability: Optional[float] = None

        if hasattr(self.model, "predict_proba"):

            proba_values = self.model.predict_proba(df)

            if proba_values.shape[1] > 1:

                probability = float(proba_values[0][1])

        return PredictionResult(
            prediction=prediction,
            label="fraud" if prediction == 1 else "not_fraud",
            probability=probability,
        )
