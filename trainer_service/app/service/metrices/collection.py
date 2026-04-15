from typing import Dict, Set
from torchmetrics import MetricCollection, Metric, MeanMetric
from torchmetrics.classification import (
    Accuracy, Recall, Precision, F1Score, ConfusionMatrix,
    AUROC, Specificity, CohenKappa
)
from torch import device

from app.api.schemes import MetricesParams, MetricesParamCollections
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
    metric_params: MetricesParams,
    num_classes: int,
    device: device,
    prefix: str,
    include_loss: bool = True,
) -> MetricCollection:
    """
    Создание коллекции метрик для классификации
    
    Args:
        metric_params: Название метрик и average
        num_classes: Количество классов
        device: cpu/cuda
        prefix: Префикс для названий метрик
        include_loss: Включить метрику loss
    Returns:
        MetricCollection: Коллекция метрик
    """
    
    # Создаём словарь метрик
    metrics_dict: Dict[str, Metric | MetricCollection] = {} # MetricCollection добавлен, чтоб синтаксис не ругался

    for metric_name in metric_params.metrics_list:
        if metric_name not in METRICS_REGISTRY:
            error_text = (
                f"Неизвестная метрика '{metric_name}'. "
                f"Список доступных метрик: {list(METRICS_REGISTRY.keys())}"
            )
            logger.error(error_text)
            logger.warning(f"Метрика '{metric_name}' не буедт учитываться.")
            continue

        base_metric_params = {
            'task': 'multiclass',
            'num_classes': num_classes,
            'sync_on_compute': False,
        }

        # Добавляем average для метрик, которые его поддерживают
        if metric_name in ['precision', 'recall', 'f1']:
            base_metric_params['average'] = metric_params.average

        # Получаем класс
        metric_class = METRICS_REGISTRY[metric_name]
        
        # Настраиваем обьект
        if metric_name == 'confusion_matrix':
            metrics_dict[metric_name] = metric_class(
                task='multiclass',
                num_classes=num_classes,
                normalize=None
            )
        else:
            metrics_dict[metric_name] = metric_class(**base_metric_params)
    
    if include_loss:
        metrics_dict['loss'] = MeanMetric(
            sync_on_compute=base_metric_params['sync_on_compute'],
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
        metric_param_collections: MetricesParamCollections,
        num_classes: int,
        device: device,
    ) -> Dict[str, MetricCollection]:
    """
    Создание колекций метрик для всех выборок

    Args:
        metric_param_collections: Схема параметров
        num_classes: Количество классов
        device: cpu/cuda
    
    Returns:
        Словарь с коллекциями
    """
    return {
        'train': create_classification_metrics(
            metric_params=metric_param_collections.train,
            num_classes=num_classes,
            device=device,
            prefix='train_'
        ),
        'val': create_classification_metrics(
            metric_params=metric_param_collections.val,
            num_classes=num_classes,
            device=device,
            prefix='val_'
        ),
        'test': create_classification_metrics(
            metric_params=metric_param_collections.test,
            num_classes=num_classes,
            device=device,
            prefix='test_',
            include_loss=False
        )      
    }