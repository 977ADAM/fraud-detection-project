from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import logging
import shap
import numpy as np

try:
    from .config import config
    from .features import add_features
    from .schema import FEATURE_SCHEMA
    from .model_repository import ModelRepository

except ImportError:
    from config import config
    from features import add_features
    from schema import FEATURE_SCHEMA
    from model_repository import ModelRepository

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    prediction: int
    label: str
    probability: Optional[float]
    shap_values: Optional[Dict[str, float]] = None

class FraudModel:

    def __init__(self, repository: ModelRepository | None = None):

        self.repository = repository or ModelRepository()
        self.repository.load()

        self.model = self.repository.model
        self.feature_schema = self.repository.feature_schema

        self._pipeline_has_feature_step = (
            hasattr(self.model, "named_steps")
            and "features" in self.model.named_steps
        )

        # Инициализация SHAP explainer для линейной модели
        try:
            clf = self.model.named_steps.get("clf")
            preprocess = self.model.named_steps.get("preprocess")
            background = np.zeros((1, preprocess.get_feature_names_out().shape[0]))
            try:
                self._explainer = shap.LinearExplainer(
                    clf,
                    background,
                    feature_perturbation="interventional",
                )
            except TypeError:
                self._explainer = shap.LinearExplainer(clf, background)
        except Exception:
            logger.warning("SHAP explainer не инициализирован")
            self._explainer = None

    def _prepare_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:

        expected_numeric_columns = list(
            self.feature_schema.get("numerical", FEATURE_SCHEMA.numerical)
        )
        expected_categorical_columns = list(
            self.feature_schema.get("categorical", FEATURE_SCHEMA.categorical)
        )
        expected_model_numeric_columns = self.feature_schema.get("model_numerical")
        engineered = list(self.feature_schema.get("engineered", []))
        required_fields = expected_numeric_columns + expected_categorical_columns

        missing = [f for f in required_fields if f not in data]

        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        try:
            row: Dict[str, Any] = {}
            for col in expected_numeric_columns:
                row[col] = float(data[col])
            for col in expected_categorical_columns:
                row[col] = data[col]
            df = pd.DataFrame([row])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid input types: {e}")
        
        if df.isnull().any().any():
            raise ValueError("Dataset содержит NaN до feature engineering")

        if expected_model_numeric_columns is None:
            # Backward compatibility:
            # старые metadata могут не содержать model_numerical.
            expected_model_numeric_columns = list(
                dict.fromkeys(expected_numeric_columns + engineered)
            )
        else:
            expected_model_numeric_columns = list(expected_model_numeric_columns)

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
        shap_dict = None

        if hasattr(self.model, "predict_proba"):
            try:
                proba_values = self.model.predict_proba(df)
                if proba_values.shape[1] > 1:
                    classes = list(getattr(self.model, "classes_", []))
                    class_index = classes.index(1) if 1 in classes else proba_values.shape[1] - 1
                    probability = float(proba_values[0][class_index])
            except Exception:
                logger.exception("predict_proba failed; return prediction without probability")
                probability = None

        # SHAP explanation
        if self._explainer is not None:
            try:
                if self._pipeline_has_feature_step:
                    prepared = self.model.named_sreps["features"].transform(df)
                else:
                    prepared = df

                transformed = self.model.named_steps["preprocess"].transform(prepared)

                shap_vals = self._explainer.shap_values(transformed)

                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

                feature_names = self.model.named_steps[
                    "preprocess"
                ].get_feature_names_out()

                shap_dict = {
                    name: float(value)
                    for name, value in zip(feature_names, shap_vals[0])
                }
            except Exception:
                logger.exception("SHAP explanation failed")
                shap_dict = None

        return PredictionResult(
            prediction=prediction,
            label="fraud" if prediction == 1 else "not_fraud",
            probability=probability,
            shap_values=shap_dict,
        )
