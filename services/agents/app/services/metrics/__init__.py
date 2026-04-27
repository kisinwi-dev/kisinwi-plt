import requests
from typing import List, Dict, Any
from requests.exceptions import RequestException

from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

class Metrics():
    
    def __init__(self) -> None:
        self.url = config_url.METRICS_URL

    def get_metrics(
        self,    
        task_id: str
    ) -> List[Dict[str, Any]]:
        """Получение метрик из сервиса метрик"""
        try:
            if not self._task_metrics_exists(task_id):
                return [{"Error": "Задачи в сервисе метрик не существует"}]
            
            logger.debug("Поиск метрик...")
            response = requests.get(f"{self.url}/metrics/task/{task_id}")
            response.raise_for_status()
            logger.debug(f"✅ Метрики для задачи ({task_id}) найдены")

            return response.json()["metrics"]
        except RequestException as e:
            logger.error(f"Ошибка при выполнении поиска метрик: {e}")
            raise

    def _task_metrics_exists(
        self, 
        task_id: str
    ) -> bool:
        try:
            response = requests.get(f"{self.url}/metrics/task/{task_id}/exists")
            response.raise_for_status()
            logger.debug(f"✅ Искомая задача ({task_id}) найдена в сервисе метрик")
            if response.json()["exists"]:
                return True
            else:
                return False
        except RequestException as e:
            logger.error(f"Ошибка при выполнении поиска метрик: {e}")
            return False


metrics = Metrics()