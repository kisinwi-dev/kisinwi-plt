import requests
from typing import Dict, Any, List
from crewai.tools import tool
from pydantic import BaseModel

from .utils import *
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

    def health(
        self,
    ) -> dict:
        try:
            # Отправляем POST запрос
            response = self.session.get(
                f"{self.URL}/info/health",
                timeout=30
            )
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            return response.json()
        
        except requests.RequestException as e:
            logger.error(f"Ошибка HTTP при обращении к сервису моделей: {e}")
            return {
                "status": "dead"
            }
        except Exception as e:
            logger.error(f"Ошибка при обращении к сервису моделей: {e}")
            return {
                "status": "dead"
            }

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
        """Отправить JSON для запуска тренировки модели"""
        try:
            
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
        
        except requests.RequestException as e:
            logger.error(f"Ошибка HTTP при отправке задачи: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при отправке задачи в сервис задач: {e}")
            raise

    def update_model(
        self,
        model_id: str,
        train_params: dict | str
    ):
        """Отправить JSON для запуска тренировки модели"""
        try:
            # Парсим JSON если это строка
            params = {}
            params["train_params"] = parse_in_json(train_params)

            logger.debug(f"Jsons отправляемый в сервис моделей: \n{params}")

            # Отправляем POST запрос
            response = self.session.patch(
                f"{self.URL}/models/{model_id}",
                json=params,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Проверяем статус ответа
            response.raise_for_status()
            response.json()["model_id"]
        
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

# ============= ИНСТРУМЕНТЫ ДЛЯ АГЕНТОВ ================

@tool("GetMLModelsInfo")
@handle_errors
def get_ml_models_info(ml_model_id: str) -> dict:
    """
    НАЗНАЧЕНИЕ: Получить полную информацию об ОДНОЙ ML модели по её ID.
    
    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать детали конкретной модели
    - Когда у вас есть один ID модели
    
    ВХОДНЫЕ ДАННЫЕ:
    - ml_model_id (str): Уникальный идентификатор модели.
      Пример: "f0536964-7950-4087-aa93-91bd50d835be"
    
    ВОЗВРАЩАЕТ:
    - dict с информацией о модели:
    
    ПРИМЕР ВЫЗОВА:
    get_ml_models_info("f0536964-7950-4087-aa93-91bd50d835be")
    """
    logger.info(f"🔧 GetMLModelsInfo вызван для модели: {ml_model_id}")
    return get_json(f"/models/{ml_model_id}")

@tool("GetAllMLModelsInfo")
@handle_errors
def get_all_ml_models_info(ml_models_id: List[str]) -> dict:
    """
    НАЗНАЧЕНИЕ: Получить информацию о НЕСКОЛЬКИХ ML моделях по списку ID.
    
    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно сравнить несколько моделей
    - Когда нужно получить информацию о 2+ моделях за один вызов
    - Основной инструмент для анализа и сравнения моделей
    
    ВХОДНЫЕ ДАННЫЕ:
    - ml_models_id (List[str]): Список ID моделей.
      Пример: ["f0536964-7950-4087-aa93-91bd50d835be", "9b5cddc4-6992-471d-b423-4de7be7c2b91"]
    
    ВОЗВРАЩАЕТ:
    - dict, где ключ = ID модели, значение = информация о модели:
        {
            "model_id_1": { ... },
            "model_id_2": { ... }
        }
    
    ПРИМЕР ВЫЗОВА:
    get_all_ml_models_info([
        "f0536964-7950-4087-aa93-91bd50d835be",
        "9b5cddc4-6992-471d-b423-4de7be7c2b91"
    ])

    🔹 ВАЖНО:
    - Этот инструмент предпочтительнее, чем многократный вызов GetMLModelsInfo
    - Используй его для сравнения моделей
    """
    logger.info(f"🔧 GetAllMLModelsInfo вызван для моделей: {ml_models_id}")
    info_models = {}

    for id in ml_models_id:
        info_models[id] = get_json(f"/models/{id}")
    return info_models
