from typing import Any, Dict, List

import torch
from torch import nn
from pydantic import ValidationError

from app.api.schemas import TaskParams
from app.core.models import get_models_type_name
from app.core.datas.augmentations import ALLOWED_TRANSFORMS
from app.service.metrices.collection import METRICS_REGISTRY
from .setup_training import get_optimizers, get_schedulers


def validate_task_params(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Валидация конфигурации обучения.

    Сначала проверяет конфигурацию по схеме TaskParams, затем — значения
    по реестрам: модели timm, функции потерь torch.nn, оптимизаторы,
    планировщики, метрики, трансформации и доступность устройства.

    Args:
        config: конфигурация обучения (JSON)

    Returns:
        Список ошибок вида {'loc': ..., 'msg': ...}. Пустой список — конфигурация валидна.
    """
    errors: List[Dict[str, str]] = []

    # Проверка по схеме
    try:
        params = TaskParams.model_validate(config)
    except ValidationError as e:
        for err in e.errors():
            loc = '.'.join(str(part) for part in err['loc'])
            errors.append({'loc': loc, 'msg': err['msg']})
        return errors

    # Модель
    model_type = params.model_params.type
    if model_type not in get_models_type_name():
        errors.append({
            'loc': 'model_params.type',
            'msg': f"Модель '{model_type}' не найдена в timm"
        })

    # Функция потерь
    loss_name = params.trainer_params.loss_fn.name
    if not hasattr(nn, loss_name):
        errors.append({
            'loc': 'trainer_params.loss_fn.name',
            'msg': f"Функция потерь '{loss_name}' не найдена в torch.nn"
        })

    # Оптимизатор
    optimizer_name = params.trainer_params.optimizer.name
    if optimizer_name not in get_optimizers():
        errors.append({
            'loc': 'trainer_params.optimizer.name',
            'msg': f"Оптимизатор '{optimizer_name}' не найден среди доступных"
        })

    # Планировщик
    scheduler_name = params.trainer_params.scheduler.name
    if scheduler_name not in get_schedulers():
        errors.append({
            'loc': 'trainer_params.scheduler.name',
            'msg': f"Планировщик '{scheduler_name}' не найден среди доступных"
        })

    # Метрики
    for split in ('train', 'val', 'test'):
        metrics_params = getattr(params.metrices_params, split)
        for metric_name in metrics_params.metrics_list:
            if metric_name not in METRICS_REGISTRY:
                errors.append({
                    'loc': f'metrices_params.{split}.metrics_list',
                    'msg': f"Неизвестная метрика '{metric_name}'"
                })

    # Метрика ранней остановки ('loss' добавляется автоматически)
    early_stop_metric = params.trainer_params.early_stop.metric_name
    if early_stop_metric != 'loss' and early_stop_metric not in METRICS_REGISTRY:
        errors.append({
            'loc': 'trainer_params.early_stop.metric_name',
            'msg': f"Неизвестная метрика '{early_stop_metric}'"
        })

    # Трансформации
    for field in ('train_transforms_config', 'val_and_test_transforms_config'):
        for i, transform_config in enumerate(getattr(params.data_loader_params, field)):
            transform_name = transform_config.get('name')
            if not transform_name:
                errors.append({
                    'loc': f'data_loader_params.{field}.{i}',
                    'msg': "У трансформации отсутствует поле 'name'"
                })
            elif transform_name not in ALLOWED_TRANSFORMS:
                errors.append({
                    'loc': f'data_loader_params.{field}.{i}.name',
                    'msg': f"Трансформация '{transform_name}' не найдена среди доступных"
                })

    # Устройство
    try:
        device = torch.device(params.device)
    except RuntimeError:
        errors.append({
            'loc': 'device',
            'msg': f"Некорректное устройство '{params.device}'"
        })
    else:
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                errors.append({
                    'loc': 'device',
                    'msg': "CUDA недоступна на этом инстансе"
                })
            elif device.index is not None and device.index >= torch.cuda.device_count():
                errors.append({
                    'loc': 'device',
                    'msg': f"Устройство 'cuda:{device.index}' не найдено, "
                           f"доступно GPU: {torch.cuda.device_count()}"
                })

    return errors
