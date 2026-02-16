import pandas as pd
import joblib
import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import mlflow
import mlflow.sklearn

try:
    from .features import add_features
    from .config import config
    from .schema import FEATURE_SCHEMA
except ImportError:
    from features import add_features
    from config import config
    from schema import FEATURE_SCHEMA

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: Union[str, Path], target_col=config.target_column):
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset пуст.")

    if target_col not in df.columns:
        raise ValueError(f"Нет колонки таргета '{target_col}'. Колонки: {list(df.columns)[:20]} ...")


    required_num_cols = FEATURE_SCHEMA.numerical
    required_cat_cols = FEATURE_SCHEMA.categorical
    required_cols = set(required_num_cols + required_cat_cols)
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(
            f"В датасете отсутствуют обязательные колонки: {missing_cols}"
        )

    y = df[target_col].astype(int).values
    unique_target_values = set(pd.Series(y).dropna().unique().tolist())
    if not unique_target_values.issubset({0, 1}):
        raise ValueError(
            f"Таргет '{target_col}' должен содержать только 0/1. "
            f"Найдено: {sorted(unique_target_values)}"
        )
    
    if len(unique_target_values) < 2:
        raise ValueError(
            f"Таргет '{target_col}' содержит только один класс: {sorted(unique_target_values)}. "
            "Для LogisticRegression требуется минимум два класса."
        )

    df = df.drop(columns=[target_col])

    X = df

    return X, y


def build_model():
    num_cols = FEATURE_SCHEMA.numerical
    cat_cols = FEATURE_SCHEMA.categorical
    engineered_cols = FEATURE_SCHEMA.engineered

    model_num_cols = num_cols + engineered_cols

    if not model_num_cols and not cat_cols:
        raise ValueError("Список признаков пуст.")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), model_num_cols),
            ("cat", OneHotEncoder(drop='first', handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", preprocess),
        (
            'clf',
            LogisticRegression(
                class_weight = config.class_weight,
                max_iter = config.max_iter,
                random_state = config.random_state,
                solver = config.solver,
            ),
        )
    ])

    return pipeline, num_cols, model_num_cols, cat_cols, engineered_cols

def save_model(
        model,
        metrics: dict,
        params: dict,
        feature_schema: dict,
        dataset_id: str,
        name: str,
        version: str,
        base_dir: Optional[Union[str, Path]] = None) -> Path:
    
    root_dir = config.model_base_path if base_dir is None else config.resolve_path(base_dir)
    out_dir = root_dir / name / version
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / config.model_file_name
    joblib.dump(model, model_path)

    metadata = {
        "name": name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "params": params,
        "metrics": metrics,
        "python": __import__("sys").version,
        "sklearn": __import__("sklearn").__version__,
        "feature_schema": feature_schema,
    }
    
    (out_dir / config.metadata_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path

def main():

    data_path = config.dataset_path
    logger.info(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset не найден по этому пути: {data_path}")

    X, y = load_data(data_path)

    if pd.isnull(X).any().any():
        raise ValueError("В данных есть NaN перед обучением.")

    stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=stratify_target,
        random_state=config.random_state,
    )

    pipeline, raw_num_cols, model_num_cols, cat_cols, engineered_cols = build_model()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    if len(set(y_test)) > 1 and hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None

    mlflow.set_experiment("fraud_detection")
    with mlflow.start_run(run_name=f"{config.name}_{config.version}"):

        # Логируем параметры
        mlflow.log_params({
            "max_iter": config.max_iter,
            "solver": config.solver,
            "class_weight": config.class_weight,
            "test_size": config.test_size,
            "random_state": config.random_state,
        })
        mlflow.log_metric("accuracy", acc)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_dict(
            {
                "numerical": raw_num_cols,
                "model_numerical": model_num_cols,
                "categorical": cat_cols,
                "engineered": engineered_cols,
            },
            "feature_schema.json"
        )
        mlflow.log_dict(report, "classification_report.json")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name="fraud_model"
        )

    logger.info("Classification report:")
    logger.info(json.dumps(report, indent=2))
    logger.info("Confusion matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    logger.info(f"Accuracy: {acc:.4f}")
    
    if roc_auc is not None:
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
    else:
        logger.info("ROC-AUC: недоступен")

    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        feature_names_list = list(feature_names)
    except Exception:
        logger.exception("Не удалось извлечь имена признаков из preprocess.")
        feature_names_list = []

    output_dir = save_model(
        model=pipeline,
        metrics={
            "accuracy": acc,
            "roc_auc": roc_auc,
        },
        params={
            "max_iter": config.max_iter,
            "random_state": config.random_state,
            "test_size": config.test_size,
            "feature_names": feature_names_list,
        },
        feature_schema = {
            "numerical": raw_num_cols,
            "model_numerical": model_num_cols,
            "categorical": cat_cols,
            "engineered": engineered_cols,
        },
        dataset_id=config.dataset_id,
        name=config.name,
        version=config.version,
    )
    logger.info(f"Модель сохранена в: {output_dir}")

if __name__ == "__main__":
    main()
