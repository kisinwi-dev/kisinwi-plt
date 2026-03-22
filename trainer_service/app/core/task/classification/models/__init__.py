from torch import nn
from . import vgg, resnet, efficientnet, convnext, vision_transformer, swin_transformer
from .registry import REGISTRY

from app.logs import get_logger

logger = get_logger(__name__)

def get_model(
        type: str = 'Resnet',  
        *args,
        **kwargs
    ) -> nn.Module:
    logger.info("Загрузка модели")
    if type.lower() not in REGISTRY:
        logger.error(f"Неизвестный тип модели: {type}")
        raise ValueError(f"Неизвестный тип модели: {type}")
    
    cls = REGISTRY[type.lower()]
    model = cls(*args, **kwargs)
    logger.info("🍋‍🟩 Модель успешно загружена")
    return model
