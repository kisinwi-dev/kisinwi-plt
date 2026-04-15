import requests
from typing import Dict, Any, Optional
from torch import device, Tensor

from .shemas import MetricesParamCollections
from .collection import create_classification_collections

from app.config import config_domain
from app.logs import get_logger

logger = get_logger(__name__)

class MetricesClient:
    def __init__(
            self, 
            task_id: str,
            metrices_params: MetricesParamCollections,
            num_class: int,
            device: device
    ):
        self._task_id = task_id
        self._domain = config_domain.METRIC
        self._collections = create_classification_collections(
            metrices_params,
            num_class,
            device
        )
        
    def update(
            self,
            type: str,
            preds: Tensor,
            targets: Tensor,
            loss: Tensor|None = None,
    ):
        """Добавление значений для расчёта метрик"""
        self._collections[type].update(
            preds=preds,
            target=targets,
            value=loss
        )

    def compute(
            self,
            type: str,
    ):
        """Рассчёт метрик и их очистка"""
        metrics = self._collections[type].compute()
        
        logger.info(f"{type} loss: {metrics.get(f'{type}_loss', 'N/A')}")
        self._collections[type].reset()

    def log_metric(
            self, 
            metric_name: str, 
            score:  float, 
            step: int
    ):
        """
        Отправка одной метрики
        
        Args:
            metric_name: Имя метрики
            score: Значение метрики
            step: Шаг(эпоха)
        """
        url = f"{self._domain}/api/metrics/update"

        payload = {
            "task_id": self._task_id,
            "metric_name": metric_name,
            "score": score,
            "step": step
        }
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()

            logger.debug(f"Метрика {metric_name} занесена в историю (score:{score})")
        except Exception as e:
            logger.error(f"Ошибка сохранения метрики {metric_name}: {e}")


    def log_epoch_metrics(
        self,
        step: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Отправляет все метрики эпохи
        Имена метрик получают префиксы: train_, val_, test_.
        """

        # Train метрики
        for name, value in train_metrics.items():
            if value is not None:
                self.log_metric(f"train_{name}", value.item(), step)

        # Validation метрики
        for name, value in val_metrics.items():
            if value is not None:
                self.log_metric(f"val_{name}", value.item(), step)

        # Test метрики (если есть)
        if test_metrics:
            for name, value in test_metrics.items():
                if value is not None:
                    self.log_metric(f"test_{name}", value.item(), step)