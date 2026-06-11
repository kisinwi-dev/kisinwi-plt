from typing import Dict
from torchmetrics import MetricCollection, Metric, MeanMetric
from torchmetrics.classification import (
    Accuracy, Recall, Precision, F1Score, ConfusionMatrix,
    AUROC, Specificity, CohenKappa
)
from torch import device

from app.logs import get_logger

logger = get_logger(__name__)

# Доступные метрики
METRICS_REGISTRY = {
    'accuracy': Accuracy,
    'precision': Precision,
    'recall': Recall,
    'f1': F1Score,
    'confusion_matrix': ConfusionMatrix,
    'auroc': AUROC,
    'specificity': Specificity,
    'kappa': CohenKappa,
}

def create_classification_metrics(
    num_classes: int,
    device: device,
    prefix: str,
    include_loss: bool = True,
) -> MetricCollection:
    """
    Создание коллекции метрик для классификации.

    Считаются все скалярные метрики из METRICS_REGISTRY (confusion_matrix
    не включается: поэпоховый канал принимает только скаляры, матрица
    уходит через class report после теста).

    Args:
        num_classes: Количество классов
        device: cpu/cuda
        prefix: Префикс для названий метрик
        include_loss: Включить метрику loss
    Returns:
        MetricCollection: Коллекция метрик
    """

    # Создаём словарь метрик
    metrics_dict: Dict[str, Metric | MetricCollection] = {} # MetricCollection добавлен, чтоб синтаксис не ругался

    for metric_name, metric_class in METRICS_REGISTRY.items():
        if metric_name == 'confusion_matrix':
            continue

        metric_params = {
            'task': 'multiclass',
            'num_classes': num_classes,
            'sync_on_compute': False,
        }

        # Добавляем average для метрик, которые его поддерживают
        if metric_name in ['precision', 'recall', 'f1']:
            metric_params['average'] = 'macro'

        metrics_dict[metric_name] = metric_class(**metric_params)

    if include_loss:
        metrics_dict['loss'] = MeanMetric(
            sync_on_compute=False,
            nan_strategy='ignore'
        )

    # Создание коллекции
    collection = MetricCollection(
        metrics_dict,
        prefix=prefix
    )

    for metric in collection.keys():
        logger.debug(f'✅ Создана метрика: {metric}')


    # Переносим на устройство
    if device:
        collection = collection.to(device)

    return collection

def create_classification_collections(
        num_classes: int,
        device: device,
    ) -> Dict[str, MetricCollection]:
    """
    Создание колекций метрик для всех выборок

    Args:
        num_classes: Количество классов
        device: cpu/cuda

    Returns:
        Словарь с коллекциями
    """
    # train/val/test считают один и тот же полный набор метрик —
    # каждая метрика сравнима между выборками для контроля переобучения
    return {
        'train': create_classification_metrics(
            num_classes=num_classes,
            device=device,
            prefix='train_'
        ),
        'val': create_classification_metrics(
            num_classes=num_classes,
            device=device,
            prefix='val_'
        ),
        'test': create_classification_metrics(
            num_classes=num_classes,
            device=device,
            prefix='test_',
            include_loss=False
        )
    }
