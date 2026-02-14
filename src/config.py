from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

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
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    def resolve_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def model_base_path(self) -> Path:
        return self.resolve_path(self.model_base_dir)

    @property
    def data_base_path(self) -> Path:
        return self.resolve_path(self.data_base_dir)

    @property
    def model_dir(self) -> Path:
        return self.model_base_path / self.name / self.version

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.pkl"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_name

    @property
    def dataset_path(self) -> Path:
        return self.data_base_path / "dataset.csv"

config = Config()
