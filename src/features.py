import pandas as pd
from typing import Tuple, List

NUMERICALS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CATEGORICALS = ["type"]
ENGINEERED = ["balanceDiffOrig", "balanceDiffDest"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет инженерные признаки для fraud detection.

    Ожидает:
        oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest
    """
    df = df.copy()
    
    # требуемые столбцы
    required_cols = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df.drop(
        ["step", "nameOrig", "nameDest", "isFlaggedFraud"],
        axis=1,
        errors="ignore",
        inplace=True,
    )

    return df

def all_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Возвращает список числовых и категориальных признаков
    с учетом инженерных признаков.
    """
    return NUMERICALS + ENGINEERED, CATEGORICALS
