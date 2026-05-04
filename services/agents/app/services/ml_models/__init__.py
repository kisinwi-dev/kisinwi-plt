import requests
from pydantic import BaseModel

from app.logs import get_logger
from app.config import config_url
from app.services.tools import parse_in_json

logger = get_logger(__name__)

class ModelMeta(BaseModel):
    model_type: str
    description: str

class MLModels():
    def __init__(self) -> None:
        self.URL = config_url.ML_MODELS_URL
        self.session = requests.Session()

    def create_model(
        self,
        name: str,
        model_type: str,
        description: str,
        classes: list,
        train_params: dict | str
    ) -> str:
        """Отправить JSON для запуска тренировки модели"""
        try:
            
            data = {
                "name": name,
                "model_type": model_type,
                "description": description,
                "classes": classes,
            }

            # Парсим JSON если это строка
            params = parse_in_json(data)
            params["train_params"] = parse_in_json(train_params)

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
        
        except requests.RequestException as e:
            logger.error(f"Ошибка HTTP при отправке задачи: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при отправке задачи в сервис задач: {e}")
            raise

    def close(self):
        self.session.close()

    def __exit__(self):
        self.close()

ml_models = MLModels()