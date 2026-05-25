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
    def create_model(
        self,
        name: str,
        version: int,
        model_type: str,
        description: str,
        classes: list,
        dataset_id: str,
        dataset_version_id: str,
        train_params: dict | str
    ) -> str:
        """
        Отправить JSON для запуска тренировки модели
        
        Returns:
            str: id созданной модели
        """
        data = {
            "name": name,
            "version": version,
            "model_type": model_type,
            "description": description,
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
            f"{self.URL}/models",
            json=params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Проверяем статус ответа
        response.raise_for_status()
        model_id = response.json()["model_id"]
        logger.debug(f"Модель создана и имеет id={model_id}")
        
        return model_id
    
    @handle_errors(ML_MODELS_URL)
    def update_model(
        self,
        model_id: str,
        train_params: dict | str
    ):
        """
        Обновление информации о модели
        """
        # Парсим JSON если это строка
        params = {}
        params["train_params"] = parse_in_json(train_params)

        logger.debug(f"Jsons отправляемый в сервис моделей: \n{params}")

        # Отправляем PATH запрос
        response = self.session.patch(
            f"{self.URL}/models/{model_id}",
            json=params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Проверяем статус ответа
        response.raise_for_status()
        response.json()["model_id"]

    def close(self):
        self.session.close()

    def __exit__(self):
        self.close()

ml_models_client = MLModelsClient()
