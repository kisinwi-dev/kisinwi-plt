import os
import requests
from typing import Dict, Any, List
from crewai.tools import tool

from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

@tool("GetExampleJSONTrainer")
def get_example_run_config_trainer_json() -> Dict[str, Any]:
    """Получить пример JSON для запуска тренировки модели"""
    try:
        resp = requests.get(f"{config_url.TRAINER_URL}/info/example_config")
        data = resp.json()
        return data
    except Exception as e:
        logger.error("Не удалось получить пример конфигураций обучения")
        return {"error": e}

@tool("GetAllAvailableModels")
def get_type_and_name_models(filter: str) -> List[str]:
    """
    Получить все имеющиеся модели в распоряжении.

    Args:
        filter - фильтр моделей(Пример: "*resnet*" - поиск всех моделей, 
    в которых есть слово resnet)
    """
    try:
        params={}
        params["filter"] = filter

        resp = requests.get(
            f"{config_url.TRAINER_URL}/info/ml_models",
            params=params
        )
        data = resp.json()
        return data
    except Exception as e:
        logger.error("Не удалось получить информацию об имеющихся моделях")
        return [f'Ошибка: {e}']

@tool("GetInfoDevice")
def get_info_device() -> Dict[str, Any]:
    """Получить информацию о технических возможностях обучения"""
    try:
        resp = requests.get(f"{config_url.TRAINER_URL}/info/device")
        data = resp.json()
        return data
    except Exception as e:
        logger.error("Не удалось получить информацию об устройствах")
        return {"error": e}