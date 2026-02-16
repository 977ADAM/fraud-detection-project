import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Union

logger = logging.getLogger(__name__)

ALLOWED_TRANSACTION_TYPES: Tuple[str, ...] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

# Типы, требующие проверки баланса отправителя
DEBIT_TRANSACTION_TYPES: Tuple[str, ...] = (
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
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
    model_file_name: str = "model.pkl"
    dataset_file_name: str = "dataset.csv"
    target_column: str = "isFraud"
    metadata_name: str = "metadata.json"
    log_level: str = "INFO"
    json_logs: bool = False
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

    def __post_init__(self):

        if self.solver not in {"lbfgs", "liblinear", "saga"}:
            raise ValueError(f"Unsupported solver: {self.solver}")
        
        if self.solver == "liblinear" and self.class_weight == "balanced":
            logger.warning("liblinear + balanced может вести к нестабильной сходимости")

        if not 0 < self.test_size < 1:
            raise ValueError("test_size должен быть в диапазоне (0,1)")

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
        return self.model_dir / self.model_file_name

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_name

    @property
    def dataset_path(self) -> Path:
        return self.data_base_path / self.dataset_file_name

config = Config()


if __name__ == "__main__":

    print(config.dataset_path)
