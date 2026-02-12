import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    return pipeline

def main():

    X, y = load_data('./data/dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipeline = build_model()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(pipeline.score(X_test, y_test) * 100)

    joblib.dump(pipeline, 'fraud_detection_pipeline.pkl')

if __name__ == "__main__":
    main()
