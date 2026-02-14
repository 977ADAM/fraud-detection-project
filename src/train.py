import pandas as pd
import joblib
import json

from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.features import add_features, all_feature_columns
from src.config import config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path: str, target_col=config.target_column):
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset пуст.")

    if target_col not in df.columns:
        raise ValueError(f"Нет колонки таргета '{target_col}'. Колонки: {list(df.columns)[:20]} ...")

    y = df[target_col].astype(int).values

    df = df.drop(columns=[target_col])

    df = add_features(df)

    X = df

    return X, y


def build_model():
    num_cols, cat_cols, engineered_cols = all_feature_columns()

    num_cols = num_cols + engineered_cols

    if not num_cols and not cat_cols:
        raise ValueError("Список признаков пуст.")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop='first', handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
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

    return pipeline, num_cols, cat_cols, engineered_cols

def save_model(
        model,
        metrics: dict,
        params: dict,
        feature_schema: dict,
        dataset_id: str,
        name: str,
        version: str,
        base_dir: str = config.model_base_dir) -> Path:
    
    out_dir = Path(base_dir) / name / version
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    joblib.dump(model, model_path)

    metadata = {
        "name": name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "params": params,
        "metrics": metrics,
        "python": __import__("sys").version,
        "feature_schema": feature_schema,
    }
    
    (out_dir / config.metadata_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

def main():

    data_path = Path(config.data_base_dir) / "dataset.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset не найден по этому пути: {data_path}")

    X, y = load_data(str(data_path))

    if pd.isnull(X).any().any():
        raise ValueError("В данных есть NaN перед обучением.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    pipeline, num_cols, cat_cols, engineered_cols = build_model()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if len(set(y_test)) > 1 and hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None

    logger.info("Classification report:")
    logger.info(classification_report(y_test, y_pred))
    logger.info("Confusion matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    logger.info(f"Accuracy: {acc:.4f}")
    
    if roc_auc is not None:
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
    else:
        logger.info("ROC-AUC: недоступен")

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

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
            "feature_names": list(feature_names),
        },
        feature_schema = {
            "numerical": num_cols,
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
