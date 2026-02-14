from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ALLOWED_TRANSACTION_TYPES: Tuple[str, ...] = (
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "DEPOSIT",
)

# Типы, требующие проверки баланса отправителя
DEBIT_TRANSACTION_TYPES: Tuple[str, ...] = (
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
)

ENGINEERED = ["balanceDiffOrig", "balanceDiffDest"]

@dataclass(frozen=True)
class Config:
    version: str = "1.0.0"
    dataset_id: str = "fraud:v1"
    name: str = "fraud"
    max_iter: int = 2000
    random_state: int = 42
    test_size: float = 0.3
    class_weight: str = "balanced"
    solver: str = "lbfgs"
    model_base_dir: str = "models"
    data_base_dir: str = "data"
    target_column: str = "isFraud"
    metadata_name: str = "metadata.json"

    @property
    def model_dir(self) -> Path:
        return Path(self.model_base_dir) / self.name / self.version

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_name

config = Config()