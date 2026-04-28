from typing import List, Dict, Any
from crewai.tools import tool

from .utils import get_json, handle_errors
from app.logs import get_logger

logger = get_logger(__name__)

@tool("GetMetricsForTask")
@handle_errors
def get_metrics(   
    task_id: str
) -> List[Dict[str, Any]]:
    """
    Получение метрик из сервиса метрик.
    
    Args:
        task_id: Id задачи
    
    Returns:
        Dict с метриками или ошибкой
    """
    json = get_json(f"/metrics/task/{task_id}")
    return json["metrics"]
    

@handle_errors
def task_metrics_exists(
    task_id: str
) -> bool:
    """
    Проверка существования метрик для задачи task_id
    
    Args:
        task_id: Id задачи
    
    Returns:
        bool или ошибка
    
    *True - метрики для task_id существуют, в противном случае отсутвуют
    """
    json = get_json(f"/metrics/task/{task_id}/exists")
    if json["exists"] is True:
        return True
    else:
        return False
    