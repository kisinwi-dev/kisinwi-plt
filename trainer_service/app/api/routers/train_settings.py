from typing import Dict, Any
from fastapi import APIRouter, Query, HTTPException

from app.core.models import get_models_type_name
from app.service.tasker import TaskParams

routers = APIRouter()

@routers.get("/get_example_config")
async def get_example_config() -> Dict[str, Any]:
    """
    Возвращает пример JSON конфигурации для обучения модели
    """
    return TaskParams.model_json_schema()


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
