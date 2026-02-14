from pathlib import Path
from typing import Dict, Any, Optional

from dataclasses import dataclass
import joblib
import pandas as pd
import json

from src.features import add_features

from src.config import config, DEBIT_TRANSACTION_TYPES, ALLOWED_TRANSACTION_TYPES

DEFAULT_MODEL_PATH = (
    Path(__file__).parent
    / config.model_base_dir
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

        metadata_path = model_path.parent / config.metadata_name

        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())

            if self.metadata.get("version") != config.version:
                raise ValueError(
                    f"Model version mismatch: "
                    f"{self.metadata.get('version')} != {config.version}"
                )
        else:
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        if not model_path.exists():

            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)


        if not hasattr(self.model, "named_steps") or "preprocess" not in self.model.named_steps:
            raise ValueError("Model does not contain 'preprocess' step")

        preprocess = self.model.named_steps["preprocess"]

        if not hasattr(preprocess, "get_feature_names_out"):
            raise ValueError("Preprocess step does not expose feature names")

        self.expected_features = preprocess.get_feature_names_out()

        self.feature_schema = self.metadata.get("feature_schema")
        if not self.feature_schema:
            raise ValueError("Feature schema missing in metadata")



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


        tx_type = df["type"].iloc[0]

        if tx_type not in ALLOWED_TRANSACTION_TYPES:
            raise ValueError(
                f"Unsupported transaction type: {tx_type}. "
                f"Allowed types: {ALLOWED_TRANSACTION_TYPES}"
            )
        

        expected_raw_columns = (
            self.feature_schema["numerical"]
            + self.feature_schema["categorical"]
        )

        missing_cols = [
            col for col in expected_raw_columns if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Input schema mismatch. Missing columns: {missing_cols}"
            )
        # Pipeline уже гарантирует корректность.
        # ML модель должна принимать данные, а не блокировать их.

        return df

    def predict(self, data: Dict[str, Any]) -> PredictionResult:

        df = self._prepare_dataframe(data)

        try:
            prediction = int(self.model.predict(df)[0])
        except Exception as e:
            raise ValueError(f"Inference failed: {e}")
        
        probability = None

        if hasattr(self.model, "predict_proba"):
            proba_values = self.model.predict_proba(df)
            if proba_values.shape[1] > 1:
                probability = float(proba_values[0][1])

        return PredictionResult(
            prediction=prediction,
            label="fraud" if prediction == 1 else "not_fraud",
            probability=probability,
        )
