import timm
import torch.nn as nn
from typing import List

from app.api.schemes import ModelParams
from app.logs import get_logger

logger = get_logger(__name__)


def get_model(
    params: ModelParams,
    num_classes: int,
) -> nn.Module:
    """
    Загрузка модели через timm
    
    Args:
        type: Имя модели (например: 'resnet50')
        num_classes: Количество классов
        pretrained: Использовать предобученные веса
    """
    logger.info(f"Загрузка модели: {params.type}, classes={num_classes}")
    
    try:
        model = timm.create_model(
            params.type,
            pretrained=params.pretrained,
            num_classes=num_classes,
        )
        logger.info(f"✅ Модель {params.type} загружена")
        return model
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {params.type}: {e}")
        raise


def get_models_type_name(
    filter: str | None = None
) -> List[str]:
    """
    Получение списка всех моделей

    Args:
        filter: фильтрация для поиска моделей
    """
    if filter:
        all_models = timm.list_models(filter)
    else:
        all_models = timm.list_models()
    
    return sorted(all_models)