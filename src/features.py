import pandas as pd
from typing import Tuple, List

try:
    from .schema import FEATURE_SCHEMA
except ImportError:
    from schema import FEATURE_SCHEMA


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
    required_cols = FEATURE_SCHEMA.numerical

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Отсутствуют обязательные колонки: {missing}. "
            f"Доступные колонки: {list(df.columns)}"
        )
    
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Колонка {col} должна быть числовой")

    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df = df.drop(columns=FEATURE_SCHEMA.drop_columns, errors="ignore")

    return df
