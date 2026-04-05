import requests
from typing import Dict, Any, Optional, Union, List
from app.logs import get_logger

logger = get_logger(__name__)

class MetricsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def log_metric(self, task_id: str, metric_name: str, value: Union[float, List[float]], step: int):
        """Отправка одной метрики"""
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
            logger.debug(f"(шаг={step}) Метрика {metric_name} занесена в историю")
        except Exception as e:
            logger.error(f"Ошибка сохранения метрики {metric_name}: {e}")


    def log_epoch_metrics(
        self,
        task_id: str,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Отправляет все метрики эпохи
        Имена метрик получают префиксы: train_, val_, test_.
        """

        # __WARNING__ НУЖНО УЧЕСТЬ ЧТО VALUE МОЖЕТ БЫТЬ В ФОРМАТЕ  tensor(0.9873, device='cuda:0')

        # Train метрики
        for name, value in train_metrics.items():
            if value is not None:
                self.log_metric(task_id, f"train_{name}", value.item(), epoch)

        # Validation метрики
        for name, value in val_metrics.items():
            if value is not None:
                self.log_metric(task_id, f"val_{name}", value.item(), epoch)

        # Test метрики (если есть)
        if test_metrics:
            for name, value in test_metrics.items():
                if value is not None:
                    self.log_metric(task_id, f"test_{name}", value.item(), epoch)

import os

METRIC_API = "http://" + os.getenv("METRICS_SERVICE", "localhost:6310")+"/api/"

met_cl = MetricsClient(METRIC_API)