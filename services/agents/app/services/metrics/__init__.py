from typing import List, Dict, Any
from crewai.tools import tool

from .utils import get_json, handle_errors
from app.logs import get_logger

logger = get_logger(__name__)

@tool("GetMetricsForModel")
@handle_errors
def get_metrics(   
    model_id: str
) -> List[Dict[str, Any]]:
    """
    Получение метрик из сервиса метрик.
    
    Args:
        task_id: Id задачи
    
    Returns:
        Dict с метриками или ошибкой
    """
    json = get_json(f"/models/{model_id}")
    return json["metrics"]
    
@tool("MetricsForModelExists")
@handle_errors
def model_metrics_exists(
    model_id: str
) -> bool:
    """
    Проверка существования метрик для задачи model_id
    
    Args:
        model_id: Id задачи
    
    Returns:
        bool или ошибка
    
    *True - метрики для model_id существуют, в противном случае отсутвуют
    """
    json = get_json(f"/models/{model_id}/exists")
    if json["exists"] is True:
        return True
    else:
        return False
    