from typing import Dict, Any
from fastapi import APIRouter, Query, HTTPException

from app.core.datas import DataLoaderParams
from app.core.models import get_models_type_name, ModelParams

routers = APIRouter()

EXAMLE_TASK_CONFIG ={
    "task_id": "cat_dog",
    "data_loader_params": DataLoaderParams.model_json_schema(),
    "model_params": ModelParams.model_json_schema(),
    "trainer_params": {
        "loss_fn_config": {
            "type": "CrossEntropyLoss",
            "params": {
                "reduction": "mean",
                "label_smoothing": 0.1
            }
        },
        "optimizer_config": {
            "type": "AdamW",
            "params": {
                "lr": 0.0001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.05,
                "amsgrad": False
            }
        },
        "scheduler_config": {
            "type": "CosineAnnealingLR",
            "params": {
                "T_max": 30,
                "eta_min": 1e-6,
                "last_epoch": -1
            }
        },
        "device": "cuda",
        "epochs": 30
    }
}


@routers.get("/get_example_config")
async def get_example_config() -> Dict[str, Any]:
    """
    Возвращает пример JSON конфигурации для обучения модели
    """
    return EXAMLE_TASK_CONFIG


@routers.get("/get_available_models")
async def get_available_models(
    filter: str | None = Query(None, description="Фильтр для поиска моделей (например: '*resnet*')")
):
    """
    Возвращает список доступных моделей
    """
    try:
        return get_models_type_name(filter)
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
