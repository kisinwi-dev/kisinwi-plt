import httpx
import requests
from app.logs import get_logger
from typing import Any, Union, List

logger = get_logger(__name__)

class MetricsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    async def log_metric(self, task_id: str, metric_name: str, value: Union[float, List[float]], step: int):
        """Отправляет одну метрику в сервис метрик"""
        url = f"{self.base_url}/metrics/update"
        payload = {
            "task_id": task_id,
            "metric_name": metric_name,
            "value": value,
            "step": step
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                logger.debug(f"Metric {metric_name} (step={step}) logged for task {task_id}")
        except Exception as e:
            # Не прерываем обучение из-за ошибки логирования
            logger.error(f"Failed to log metric {metric_name} for task {task_id}: {e}")

    def log_metric_sync(self, task_id: str, metric_name: str, value: Union[float, List[float]], step: int):
        url = f"{self.base_url}/metrics/update"
        payload = {
            "task_id": task_id,
            "metric_name": metric_name,
            "value": value,
            "step": step
        }
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to log metric {metric_name}: {e}")