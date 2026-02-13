import pandas as pd
import joblib
import json

from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.features import add_features, all_feature_columns

def load_data(path: str, target_col="isFraud"):
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Нет колонки таргета '{target_col}'. Колонки: {list(df.columns)[:20]} ...")
    
    df = add_features(df)

    y = df[target_col].astype(int).values

    X = df.drop(columns=[target_col])

    return X, y


def build_model():
    num_cols, cat_cols = all_feature_columns()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop='first', handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            n_jobs=None
            )
        )
    ])

    return pipeline

def save_model(
        model,
        metrics: dict,
        params: dict,
        dataset_id: str,
        name: str,
        version: str,
        base_dir: str = "models") -> Path:
    
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
    }
    
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir

def main():

    data_path = Path("./data/dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset не найден по этому пути: {data_path}")

    X, y = load_data(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipeline = build_model()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(pipeline.score(X_test, y_test) * 100)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    save_model(
        model=pipeline,
        metrics={"accuracy": acc},
        params={"max_iter": 1000},
        dataset_id="fraud:v1:random_state=42:test=0.3",
        name="fraud",
        version="1.0.0",
    )

if __name__ == "__main__":
    main()
