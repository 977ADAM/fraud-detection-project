import pandas as pd
from typing import Tuple, List

NUMERICALS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CATEGORICALS = ["type"]
ENGINEERED = ["balanceDiffOrig", "balanceDiffDest"]
DROP_COLUMNS = ["step", "nameOrig", "nameDest", "isFlaggedFraud"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет инженерные признаки для fraud detection.

    Требуемые колонки:
        - oldbalanceOrg
        - newbalanceOrig
        - oldbalanceDest
        - newbalanceDest
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df должен быть pandas.DataFrame")

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
        raise ValueError(
            f"Отсутствуют обязательные колонки: {missing}. "
            f"Доступные колонки: {list(df.columns)}"
        )
    
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Колонка {col} должна быть числовой")
        
    if (df[required_cols] < 0).any().any():
        raise ValueError("Баланс не может быть отрицательным")

    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    return df

def all_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Возвращает список числовых и категориальных признаков
    с учетом инженерных признаков.
    """
    return list(NUMERICALS + ENGINEERED), list(CATEGORICALS)
