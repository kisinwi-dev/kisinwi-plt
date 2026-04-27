import requests
from typing import List, Dict, Any
from crewai.tools import tool
from requests.exceptions import RequestException

from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

@tool("GetMetricsForTask")
def get_metrics(   
    task_id: str
) -> List[Dict[str, Any]]:
    """
    Получение метрик из сервиса метрик.
    
    Args:
        task_id: Id задачи
    """
    try:
        if not task_metrics_exists(task_id):
            return [{"Error": "Задачи в сервисе метрик не существует"}]
        
        logger.debug("Поиск метрик...")
        response = requests.get(f"{config_url.METRICS_URL}/metrics/task/{task_id}")
        response.raise_for_status()
        logger.debug(f"✅ Метрики для задачи ({task_id}) найдены")

        return response.json()["metrics"]
    except RequestException as e:
        logger.error(f"Ошибка при выполнении поиска метрик: {e}")
        raise

def task_metrics_exists(
    task_id: str
) -> bool:
    try:
        response = requests.get(f"{config_url.METRICS_URL}/metrics/task/{task_id}/exists")
        response.raise_for_status()
        logger.debug(f"✅ Искомая задача ({task_id}) найдена в сервисе метрик")
        if response.json()["exists"]:
            return True
        else:
            return False
    except RequestException as e:
        logger.error(f"Ошибка при выполнении поиска метрик: {e}")
        return False
