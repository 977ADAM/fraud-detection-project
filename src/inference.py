from pathlib import Path
from typing import Dict, Any, Optional

from dataclasses import dataclass
import joblib
import pandas as pd
import json

from config import config, ALLOWED_TRANSACTION_TYPES

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = config.model_path


@dataclass
class PredictionResult:
    prediction: int
    label: str
    probability: Optional[float]


class FraudModel:

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        model_path = config.resolve_path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        metadata_path = model_path.parent / config.metadata_name

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        if self.metadata.get("version") != config.version:
            raise ValueError(
                f"Model version mismatch. "
                f"Metadata version={self.metadata.get('version')} "
                f"Config version={config.version}"
            )

        self.model = joblib.load(model_path)

        self.feature_schema = self.metadata.get("feature_schema")
        if not self.feature_schema:
            raise ValueError("Feature schema missing in metadata")
        
        engineered = self.feature_schema.get("engineered")
        if engineered is None:
            raise ValueError("В схеме функций отсутствуют инженерные функции.")



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
        
        if data["type"] not in ALLOWED_TRANSACTION_TYPES:
            raise ValueError(
                f"Unsupported transaction type '{data['type']}'. "
                f"Allowed: {ALLOWED_TRANSACTION_TYPES}"
            )

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

        numeric_cols = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ]

        if (df[numeric_cols] < 0).any().any():
            raise ValueError("Отрицательные значения недопустимы")

        expected_raw_columns = (
            self.feature_schema.get("numerical", [])
            + self.feature_schema.get("categorical", [])
        )

        missing_cols = [
            col for col in expected_raw_columns if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Input schema mismatch. Missing columns: {missing_cols}"
            )
        
        unexpected_cols = [
            col for col in df.columns if col not in expected_raw_columns
        ]
        if unexpected_cols:
            logger.warning(f"Неожиданные столбцы во входных данных: {unexpected_cols}")

        logger.info(f"Столбцы фрейма данных Inference: {df.columns.tolist()}")

        return df

    def predict(self, data: Dict[str, Any]) -> PredictionResult:

        df = self._prepare_dataframe(data)

        try:
            prediction_array = self.model.predict(df)
        except Exception as e:
            logger.exception("Model inference failed")
            raise ValueError(f"Inference failed: {e}")

        if len(prediction_array) != 1:
            raise ValueError("Unexpected prediction shape")

        prediction = int(prediction_array[0])
        
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
