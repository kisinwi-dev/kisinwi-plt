import requests
from pydantic import BaseModel

from ..utils import handle_errors, parse_in_json
from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

ML_MODELS_URL = config_url.ML_MODELS['url']

class ModelMeta(BaseModel):
    model_type: str
    description: str

class MLModelsClient():
    def __init__(self) -> None:
        self.URL = ML_MODELS_URL
        self.session = requests.Session()

    @handle_errors(ML_MODELS_URL)
    def create_model_version(
        self,
        name: str,
        model_type: str,
        description: str,
        classes: list,
        dataset_id: str,
        dataset_version_id: str,
        train_params: dict | str
    ) -> dict:
        """
        Создать версию модели для запуска тренировки.

        Родительская модель ищется по имени (get-or-create), номер версии
        назначает сервис ml_models.

        Returns:
            dict: {"version_id": str, "version": int}
        """
        model_id = self._get_or_create_model(name, description)

        data = {
            "model_type": model_type,
            "classes": classes,
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id
        }

        # Парсим JSON если это строка
        params = parse_in_json(data)
        params["train_params"] = parse_in_json(train_params)

        logger.debug(f"Jsons отправляемый в сервис моделей: \n{params}")

        # Отправляем POST запрос
        response = self.session.post(
            f"{self.URL}/models/{model_id}/versions",
            json=params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        # Проверяем статус ответа
        response.raise_for_status()
        result = response.json()
        logger.debug(
            f"Создана версия {result['version']} (id={result['version_id']}) модели '{name}'"
        )

        return {"version_id": result["version_id"], "version": result["version"]}

    def _get_or_create_model(self, name: str, description: str) -> str:
        """
        Найти модель по имени или создать новую. Возвращает id модели.

        При повторном запуске пайплайна с тем же именем описание обновляется.
        """
        response = self.session.post(
            f"{self.URL}/models",
            json={"name": name, "description": description},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 409:
            # Модель уже есть — берём её id и освежаем описание
            response = self.session.get(
                f"{self.URL}/models/by-name/{name}",
                timeout=30
            )
            response.raise_for_status()
            model_id = response.json()["id"]

            patch_response = self.session.patch(
                f"{self.URL}/models/{model_id}",
                json={"description": description},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            patch_response.raise_for_status()

            logger.debug(f"Модель '{name}' уже существует, id={model_id}")
            return model_id

        response.raise_for_status()
        model_id = response.json()["model_id"]
        logger.debug(f"Модель '{name}' создана, id={model_id}")
        return model_id

    @handle_errors(ML_MODELS_URL)
    def update_version(
        self,
        version_id: str,
        train_params: dict | str
    ):
        """
        Обновление параметров обучения версии модели
        """
        # Парсим JSON если это строка
        params = {}
        params["train_params"] = parse_in_json(train_params)

        logger.debug(f"Jsons отправляемый в сервис моделей: \n{params}")

        # Отправляем PATCH запрос
        response = self.session.patch(
            f"{self.URL}/versions/{version_id}",
            json=params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        # Проверяем статус ответа
        response.raise_for_status()

    def close(self):
        self.session.close()

    def __exit__(self):
        self.close()

ml_models_client = MLModelsClient()
