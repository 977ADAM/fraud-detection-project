from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class FeatureSchema:
    numerical: List[str]
    categorical: List[str]
    engineered: List[str]
    drop_columns: List[str]

    @property
    def raw_features(self) -> List[str]:
        """Признаки, которые должны прийти на вход модели"""
        return self.numerical + self.categorical

    @property
    def model_numerical(self) -> List[str]:
        """Числовые признаки, которые идут в модель"""
        return self.numerical + self.engineered

    @property
    def model_features(self) -> List[str]:
        """Все признаки после feature engineering"""
        return self.model_numerical + self.categorical


# Глобальный контракт модели
FEATURE_SCHEMA = FeatureSchema(
    numerical=[
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ],
    categorical=[
        "type"
    ],
    engineered=[
        "balanceDiffOrig",
        "balanceDiffDest"
    ],
    drop_columns=[
        "step",
        "nameOrig",
        "nameDest",
        "isFlaggedFraud"
    ]
)
