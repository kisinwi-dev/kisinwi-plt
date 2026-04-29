import requests
from typing import Dict
from torch import device, Tensor

from .collection import create_classification_collections

from app.api.schemes import MetricesParamCollections
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
        self._url = config_domain.METRIC
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
            type_: str,
    ):
        """Рассчёт метрик и их очистка"""
        metrics = self._collections[type_].compute()
        logger.debug('Метрики расчитаны.')
        
        if type_ != 'test':
            logger.info(f"{type_} loss: {metrics.get(f'{type_}_loss', 'N/A')}")
        else: 
            logger.debug('Метрики на тестовых даннных')
            for key, value in metrics.items():
                name_metric = "".join(key.split('_')[1:])
                logger.info(f"{name_metric:^20}: {value:.5}")

        self._send_in_service(metrics)

        self._collections[type_].reset()
        logger.debug('✅ Коллекция метрик очищена')

    def _send_in_service(
            self,
            metrics: Dict[str, Tensor]
    ):
        """Отправка метрик в сервис метрик"""
        try:
            metrics_data = {
                "task_id": self._task_id,
                "metrics": [
                    {
                        "name": name,
                        "values": [value.item()] if value.numel() == 1 else value.tolist()
                    }
                    for name, value in metrics.items()
                ]
            }

            response = requests.post(
                f"{self._url}/training/adds",
                json=metrics_data,
                timeout=30
            )
            response.raise_for_status()
            logger.debug(f"✅ Метрики для задачи {self._task_id} отправлены в сервис метрик")

        except requests.RequestException as e:
            logger.error(f"😡 Не удалось передать метрики в сервис метрик: {e}")
