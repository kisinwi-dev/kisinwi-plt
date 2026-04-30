from typing import Dict, Any
from crewai.tools import tool

from .utils import *
from app.logs import get_logger

logger = get_logger(__name__)

@tool("GetExampleJSONTrainer")
@handle_errors
def get_example_run_config_trainer_json() -> Dict[str, Any]:
    """Получить пример JSON для запуска тренировки модели"""
    return get_json(f"/info/example_config")

@tool("GetAllAvailableModels")
@handle_errors
def get_type_and_name_models(filter: str) -> Dict[str, Any]:
    """
    Получить все имеющиеся модели в распоряжении.

    Args:
        filter - фильтр моделей(Пример: "*resnet*" - поиск всех моделей, 
    в которых есть слово resnet)
    """
    return get_json("/info/ml_models",params={"filter": filter})

@tool("GetInfoDevice")
@handle_errors
def get_info_device() -> Dict[str, Any]:
    """Получить информацию о технических возможностях обучения"""
    return get_json("/info/device")

@tool("GetOptimizers")
@handle_errors
def get_optimizers() -> Dict[str, Any]:
    """Получить список оптимайзеров"""
    return get_json("/info/optimizers")

@tool("GetSchedulers")
@handle_errors
def get_scheduler() -> Dict[str, Any]:
    """Получить список планировщиков"""
    return get_json("/info/schedulers")