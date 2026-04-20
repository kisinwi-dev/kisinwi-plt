from typing import Dict, Any, List
from fastapi import APIRouter, Query, HTTPException, status

from app.logs import get_logger
from app.core.utils import get_system_info as system_info
from app.core.models import get_models_type_name
from app.api.schemes import TaskParams

logger = get_logger(__name__)

routers = APIRouter(
    prefix='/info',
    tags=['Information']
)

@routers.get(
        "/example_config",
        response_model=Dict[str, Any],
        summary="Получить пример конфигураций для запуска обучения"
)
async def get_example_config() -> Dict[str, Any]:
    """
    Возвращает пример JSON конфигурации для обучения модели
    """
    try:
        config_schema = TaskParams.model_json_schema()
        return config_schema
    except Exception as e:
        logger.error(f"Ошибка при получении примера конфигруций: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось сгенерировать схему конфигурации: {e}"
        )


@routers.get(
        "/ml_models",
        response_model=List[str],
        summary="Получить список доступных ML моделей"
)
async def get_available_models(
    filter: str | None = Query(None, description="Фильтр для поиска моделей (например: '*resnet*')")
):
    """
    Возвращает список доступных моделей
    """
    try:
        models = get_models_type_name(filter)
        return models 
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить список доступных моделей: {e}"
        )

@routers.get(
        "/device",
        response_model=Dict[str, Any],
        summary="Получить информация о технических возможностях обучения"
)
async def get_system_info():
    """
    Возвращает список доступных вычислительных устройств
    """
    try:
        si = system_info()
        return si
    except Exception as e:
        logger.error(f"Ошибка при получении информации о технических возможностях обучения: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось получить список имеющихся вычислительных устройств: {e}"
        )

