from dataclasses import dataclass


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

config = Config()