import httpx
import requests
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Any, List
from torch import device, device as torch_device, Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import ConfusionMatrix, Precision, Recall, F1Score

from .collection import create_classification_collections

from app.api.schemas import EarlyStop
from app.config import config_services
from app.logs import get_logger

logger = get_logger(__name__)

METRICS_URL = config_services.METRICS['url']

# Общий async-клиент для статусов обучения (живёт всё время работы воркера)
_status_client = httpx.AsyncClient(timeout=30.0)

async def send_training_status(
    model_id: str,
    status: str
) -> bool:
    """
    Отправка статуса обучения в сервис метрик
    (in_progress / completed / failed / cancelled).

    Ошибки не пробрасываются: репортинг в metrics
    не должен ломать поток статусов tasker.
    """
    try:
        res = await _status_client.post(
            f"{METRICS_URL}/models/{model_id}/status",
            json={"status": status}
        )
        res.raise_for_status()
        logger.debug(f"✅ Статус обучения '{status}' модели {model_id} отправлен в сервис метрик")
        return True
    except httpx.HTTPError as e:
        logger.error(f"😡 Не удалось отправить статус обучения модели {model_id}: {e}")
        return False

async def send_checkpoint_info(
    model_id: str,
    epoch: int,
    metric: str,
    value: float | None,
) -> bool:
    """
    Отправка в сервис метрик информации о сохранённых весах: эпоха чекпоинта
    (лучшая по early-stop-метрике, либо финальная — тогда value = None),
    early-stop-метрика (чистое имя, подразумевается val) и её значение.

    Ошибки не пробрасываются: репортинг в metrics не должен ломать обучение.
    """
    try:
        res = await _status_client.post(
            f"{METRICS_URL}/models/{model_id}/checkpoint",
            json={"epoch": epoch, "metric": metric, "value": value}
        )
        res.raise_for_status()
        logger.debug(f"✅ Checkpoint модели {model_id} (эпоха {epoch}) отправлен в сервис метрик")
        return True
    except httpx.HTTPError as e:
        logger.error(f"😡 Не удалось отправить checkpoint модели {model_id}: {e}")
        return False

class MetricesClient:
    """
    Клиент метрик отвечает за работу с метриками.
    - Создание списка метрик и посчёт каждой метрики
    - Передача данных в сервис метрик

    Args:
        model_id: Id модели
        classes: Названия классов (количество выводится из длины списка)
        device: Устройство на котором раполагается весь pipeline обучения
        early_stop_params: Параметры для трекинга остановки
    """

    def __init__(
        self,
        model_id: str,
        classes: List[str],
        device: device,
        early_stop_params: EarlyStop = EarlyStop(),
    ):
        # Информация для расчёта метрик
        self._model_id = model_id
        self._url = config_services.METRICS['url']
        self._classes = list(classes)
        num_class = len(self._classes)

        # Обучение на GPU → метрики на CPU и параллельно (фоновый поток):
        # состояние метрик (AUROC хранит все логиты эпохи) не занимает VRAM,
        # а накопление не блокирует цикл обучения.
        # Обучение на CPU → метрики на том же устройстве и последовательно:
        # фоновый поток лишь конкурировал бы за ядра с обучением.
        self._parallel = device.type != 'cpu'
        self._metrics_device = torch_device('cpu') if self._parallel else device
        # Один воркер — порядок update/compute между батчами сохраняется
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='metrics')
            if self._parallel else None
        )
        self._pending_updates: List[Future] = []
        logger.info(
            f"Метрики считаются на {self._metrics_device.type}"
            f"{' (параллельно с обучением)' if self._parallel else ''}"
        )

        # Всегда полный набор метрик из METRICS_REGISTRY — конфигом не задаётся
        self._collections = create_classification_collections(
            num_class,
            self._metrics_device
        )

        # Отчёт по классам на test: считается всегда, независимо от метрик конфига
        # (платформа про классификацию — confusion matrix и per-class почти обязательны)
        class_report_params = {
            'task': 'multiclass',
            'num_classes': num_class,
            'sync_on_compute': False,
        }
        self._class_report_collection = MetricCollection({
            'confusion_matrix': ConfusionMatrix(
                task='multiclass',
                num_classes=num_class,
                normalize=None
            ),
            'precision': Precision(average='none', **class_report_params),
            'recall': Recall(average='none', **class_report_params),
            'f1': F1Score(average='none', **class_report_params),
        }).to(self._metrics_device)

        # Информация для определения логики досрочного окончания
        self._early_stop = early_stop_params
        self._metrics_early_stop_values = {
            f"val_{early_stop_params.metric_name}": []
        }
        # Последнее значение метрики ранней остановки (для выбора лучшей эпохи)
        self.last_early_stop_value: float | None = None

    def update(
        self,
        type: str,
        preds: Tensor,
        targets: Tensor,
        loss: Tensor|None = None,
    ):
        """
        Добавление значений для расчёта метрик.

        При обучении на GPU копирует тензоры на CPU и накапливает метрики
        в фоновом потоке — цикл обучения не ждёт расчёта.
        """
        if self._parallel:
            preds = preds.detach().to(self._metrics_device)
            targets = targets.detach().to(self._metrics_device)
            if loss is not None:
                loss = loss.detach().to(self._metrics_device)
            self._pending_updates.append(
                self._executor.submit(self._update_sync, type, preds, targets, loss)
            )
        else:
            self._update_sync(type, preds, targets, loss)

    def _update_sync(
        self,
        type: str,
        preds: Tensor,
        targets: Tensor,
        loss: Tensor|None,
    ):
        """Непосредственное накопление метрик (в потоке-воркере при GPU-обучении)"""
        self._collections[type].update(
            preds=preds,
            target=targets,
            value=loss
        )
        # На test дополнительно копим данные для отчёта по классам (loss ему не нужен)
        if type == 'test':
            self._class_report_collection.update(preds, targets)

    def _drain_pending_updates(self):
        """
        Дожидается фоновых update() перед compute: метрики эпохи должны
        быть посчитаны по всем батчам. Ошибки фоновых задач пробрасываются здесь.
        """
        for future in self._pending_updates:
            future.result()
        self._pending_updates.clear()

    def compute(
        self,
        type_: str,
    ) -> bool:
        """
        Рассчёт метрик и их очистка. Проверка актуальности обучения.
        
        Returns:
            bool: где True - обучение актуально. False - требуется остановка обучения
        """
        # Все фоновые update должны завершиться до расчёта
        self._drain_pending_updates()

        # Расчёт метрик
        metrics = self._collections[type_].compute()
        logger.debug('✅ Метрики расчитаны')

        # Проверка актуальности обучения
        relevance = self._check_relevance(type_, metrics)

        self._send_in_service(metrics)

        # очистка коллекции для дальнейших расчётов
        self._collections[type_].reset()
        logger.debug('✅ Коллекция метрик очищена')
        return relevance

    def _check_relevance(
        self,
        type_: str,
        metrics: Dict[str, Tensor]
    ) -> bool:
        """
        Проверка, нужно ли остановить обучение.

        Args:
            type_: Тип метрик train/test/train
            metrics: Словарь метрик

        Returns:
            True - продолжаем обучение
            False - останавливаем обучение
        """

        if type_ == 'train':
            logger.info(f"{type_} loss: {metrics.get(f'{type_}_loss', 'N/A')}")
            return True
        elif type_ == 'test':
            logger.info('Метрики на тестовых даннных')
            for key, value in metrics.items():
                # Не-скалярные тензоры (confusion_matrix, average='none')
                # числом не форматируются — в лог идут только скаляры
                if value.numel() != 1:
                    continue
                name_metric = "_".join(key.split('_')[1:])
                logger.info(f"{name_metric:^20}: {value.item():.5}")
            return True

        # Проверяем актуальность на валидационных метриках
        patience = self._early_stop.patience
        min_delta = self._early_stop.min_delta
        logger.info(f"{type_} loss: {metrics.get(f'{type_}_loss', 'N/A')}")

        for metric_name, history in self._metrics_early_stop_values.items():
            if metric_name not in metrics:
                logger.warning(f"⚠️ Целевая метрика {metric_name} не найдена в результатах")
                continue
            
            current_value = metrics[metric_name]
            self._metrics_early_stop_values[metric_name].append(current_value.item())
            self.last_early_stop_value = current_value.item()

            # Проверяем, достаточно ли данных
            if len(history) > patience:
                old_value = history[-patience - 1]
                delta = abs(current_value - old_value)

                logger.debug(f"Изменение за {patience} эпох: {delta:.6f})")
                if delta < min_delta:
                    logger.warning(f" РАННЯЯ ОСТАНОВКА! Метрика {metric_name} не изменилась за {patience} эпох на {min_delta}")
                    return False

        return True

    def _send_in_service(
        self,
        metrics: Dict[str, Tensor]
    ):
        """Отправка метрик в сервис метрик"""
        try:
            scalar_metrics = []
            for name, value in metrics.items():
                # Поэпоховый канал хранит только скаляры; матрицы/векторы
                # (confusion_matrix, average='none') идут через class report
                if value.numel() != 1:
                    logger.warning(
                        f"Метрика {name} не скалярная ({tuple(value.shape)}) — "
                        f"пропущена в поэпоховой отправке"
                    )
                    continue
                scalar_metrics.append({"name": name, "values": [value.item()]})

            metrics_data = {
                "model_id": self._model_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": scalar_metrics,
            }

            response = requests.post(
                f"{self._url}/models/adds",
                json=metrics_data,
                timeout=30
            )
            response.raise_for_status()
            logger.debug(f"✅ Метрики модели {self._model_id} отправлены в сервис метрик")

        except requests.RequestException as e:
            logger.error(f"😡 Не удалось передать метрики в сервис метрик: {e}")

    def send_class_report(self):
        """
        Расчёт и отправка отчёта по классам (test) в сервис метрик:
        confusion matrix и per-class precision/recall/f1/support.

        Ошибки не пробрасываются: отчёт — вспомогательный артефакт,
        его потеря не должна ронять обучение.
        """
        try:
            self._drain_pending_updates()
            computed = self._class_report_collection.compute()
            cm = computed['confusion_matrix'].int().tolist()
            precision = computed['precision'].tolist()
            recall = computed['recall'].tolist()
            f1 = computed['f1'].tolist()

            report = {
                "labels": self._classes,
                "confusion_matrix": cm,
                "per_class": [
                    {
                        "label": label,
                        "precision": precision[i],
                        "recall": recall[i],
                        "f1": f1[i],
                        # Support — число истинных примеров класса: сумма строки матрицы
                        "support": sum(cm[i]),
                    }
                    for i, label in enumerate(self._classes)
                ],
            }

            response = requests.post(
                f"{self._url}/models/{self._model_id}/class-report",
                json=report,
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"✅ Class report модели {self._model_id} отправлен в сервис метрик")

        except requests.RequestException as e:
            logger.error(f"😡 Не удалось передать class report в сервис метрик: {e}")
        finally:
            self._class_report_collection.reset()

    def close(self):
        """
        Остановка фонового потока метрик. Вызывать по окончании обучения
        (в т.ч. при ошибке), иначе воркер-процесс накапливает потоки.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            self._pending_updates.clear()
