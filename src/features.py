import pandas as pd

NUMERICALS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CATEGORICALS = ["type"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет инженерные признаки для fraud detection.

    Ожидает:
        oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest
    """
    df = df.copy()

    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df.drop(
        ["step", "nameOrig", "nameDest", "isFlaggedFraud"],
        axis=1,
        errors="ignore",
        inplace=True,
    )

    return df

def all_feature_columns():
    # после add_features появятся новые числовые колонки
    engineered = ["balanceDiffOrig", "balanceDiffDest"]
    return NUMERICALS + engineered, CATEGORICALS
