import timm
import torch.nn as nn
from typing import List

from app.logs import get_logger

logger = get_logger(__name__)


def get_model(
    type: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Загрузка модели через timm
    
    Args:
        type: Имя модели (например, 'resnet50', 'efficientnet_b3')
        num_classes: Количество классов
        pretrained: Использовать предобученные веса
    """
    logger.info(f"Загрузка модели: {type}, classes={num_classes}")
    
    try:
        model = timm.create_model(
            type,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        logger.info(f"✅ Модель {type} загружена")
        return model
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {type}: {e}")
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