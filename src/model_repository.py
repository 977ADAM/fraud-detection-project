from pathlib import Path
from typing import Any, Dict
import logging
import json
import joblib

try:
    from .config import config
except ImportError:
    from config import config

logger = logging.getLogger(__name__)

class ModelRepository:
    """
    Отвечает за:
    - загрузку модели
    - чтение metadata
    - проверку версии
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = config.resolve_path(
            model_path or config.model_path
        )

        self.metadata_path = self.model_path.parent / config.metadata_name

        self._model = None
        self._metadata: Dict[str, Any] | None = None

    def load(self):
        self._load_metadata()
        self._validate_metadata()
        self._load_model()

    def _load_metadata(self):
        """
        Загрузка метаданных
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Метаданные не найдены: {self.metadata_path}"
            )

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    def _validate_metadata(self):
        """
        Валидация метаданных
        """
        if self._metadata is None:
            raise RuntimeError("Метаданные не загружены")
        if not isinstance(self._metadata, dict):
            raise ValueError("Некорректный формат metadata: ожидается JSON-объект")

        if self._metadata.get("version") != config.version:
            raise ValueError(
                f"Несоответствие версий модели. "
                f"Metadata={self._metadata.get('version')} "
                f"Config={config.version}"
            )

        if "feature_schema" not in self._metadata:
            raise ValueError("Отсутствуют метаданные feature_schema")
        if not isinstance(self._metadata["feature_schema"], dict):
            raise ValueError("Некорректный формат feature_schema в metadata")

    def _load_model(self):
        """
        Загрузка модели
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {self.model_path}"
            )

        self._model = joblib.load(self.model_path)

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Модель не загружена")
        return self._model

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            raise RuntimeError("Метаданные не загружены")
        return self._metadata

    @property
    def feature_schema(self) -> Dict[str, Any]:
        return self.metadata["feature_schema"]
