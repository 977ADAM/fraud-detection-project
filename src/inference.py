from pathlib import Path
from typing import Dict, Any, Optional

from dataclasses import dataclass
import joblib
import pandas as pd
import json

try:
    from .config import config
    from .features import add_features
    from .schema import FEATURE_SCHEMA
except ImportError:
    from config import config
    from features import add_features
    from schema import FEATURE_SCHEMA

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
        
        if "feature_schema" not in self.metadata:
            raise ValueError("Metadata does not contain feature_schema")

        self.feature_schema = self.metadata["feature_schema"]
        
        if self.metadata.get("version") != config.version:
            raise ValueError(
                f"Model version mismatch. "
                f"Metadata version={self.metadata.get('version')} "
                f"Config version={config.version}"
            )

        self.model = joblib.load(model_path)
        self._pipeline_has_feature_step = (
            hasattr(self.model, "named_steps") and "features" in self.model.named_steps
        )

    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:

        required_fields = (
            FEATURE_SCHEMA.numerical
            + FEATURE_SCHEMA.categorical
        )

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

        expected_numeric_columns = self.feature_schema.get("numerical", [])
        expected_model_numeric_columns = self.feature_schema.get("model_numerical")
        engineered = self.feature_schema.get("engineered", [])
        expected_categorical_columns = self.feature_schema.get("categorical", [])

        if expected_model_numeric_columns is None:
            # Backward compatibility:
            # старые metadata могут не содержать model_numerical.
            expected_model_numeric_columns = list(
                dict.fromkeys(expected_numeric_columns + engineered)
            )
        expected_raw_columns = [
            col for col in expected_numeric_columns if col not in engineered
        ] + expected_categorical_columns

        missing_cols = [
            col for col in expected_raw_columns if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Input schema mismatch. Missing columns: {missing_cols}"
            )

        if self._pipeline_has_feature_step:
            expected_model_columns = list(dict.fromkeys(expected_raw_columns))
        else:
            try:
                # Добавляем engineered признаки для моделей без отдельного шага features.
                df = add_features(df)
            except Exception as e:
                raise ValueError(f"Feature engineering failed: {e}")

            expected_model_columns = list(
                dict.fromkeys(expected_model_numeric_columns + expected_categorical_columns)
            )
        missing_model_cols = [
            col for col in expected_model_columns if col not in df.columns
        ]
        if missing_model_cols:
            raise ValueError(
                f"Input schema mismatch after feature engineering. Missing columns: {missing_model_cols}"
            )

        unexpected_cols = [
            col for col in df.columns if col not in expected_model_columns
        ]
        if unexpected_cols:
            logger.warning(f"Неожиданные столбцы во входных данных: {unexpected_cols}")

        df = df[expected_model_columns]

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
                classes = list(getattr(self.model, "classes_", []))
                class_index = classes.index(1) if 1 in classes else proba_values.shape[1] - 1
                probability = float(proba_values[0][class_index])

        return PredictionResult(
            prediction=prediction,
            label="fraud" if prediction == 1 else "not_fraud",
            probability=probability,
        )
