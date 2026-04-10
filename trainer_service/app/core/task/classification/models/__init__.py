from collections import defaultdict
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

def get_models_type_name()-> dict:
    dtn = defaultdict(list)

    list_models_module = list([vgg, resnet, efficientnet, convnext, vision_transformer, swin_transformer])

    for module in list_models_module:
        model_type = module.__name__.split('.')[-1]
        model_names = list(module.model_mapping.keys())
        for model_name in model_names:
            dtn[model_type].append(model_name)
    
    return dtn