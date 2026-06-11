from typing import Dict, Any, List
from fastapi import APIRouter, Query, HTTPException, status

from app.logs import get_logger
from app.config import config_services
from app.core.utils import get_system_info as system_info
from app.core.utils import get_schedulers as schedulers, get_optimizers as optimizers
from app.core.models import get_models_type_name
from app.service.metrices.collection import METRICS_REGISTRY
from app.core.datas.augmentations import ALLOWED_TRANSFORMS
from app.api.schemas import TaskParams, HealthResponse

logger = get_logger(__name__)

router = APIRouter(
    prefix='/info',
    tags=['Information']
)

@router.get(
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
        logger.error(f"Ошибка при получении примера конфигураций: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось сгенерировать схему конфигурации"
        )


@router.get(
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
        logger.error(f"Ошибка при получении списка моделей: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить список доступных моделей"
        )

@router.get(
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
        logger.error(f"Ошибка при получении информации о технических возможностях обучения: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить список имеющихся вычислительных устройств"
        )

@router.get(
        "/optimizers",
        response_model=List[str],
        summary="Получить список оптимизаторов"
)
async def get_optimizers():
    """
    Возвращает список доступных оптимизаторов устройств
    """
    try:
        return optimizers()
    except Exception as e:
        logger.error(f"Ошибка при получении списка оптимизаторов: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить список имеющихся оптимизаторов"
        )

@router.get(
        "/schedulers",
        response_model=List[str],
        summary="Получить список планировщиков"
)
async def get_schedulers():
    """
    Возвращает список доступных планировщиков
    """
    try:
        return schedulers()
    except Exception as e:
        logger.error(f"Ошибка при получении списка планировщиков: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить список имеющихся планировщиков"
        )

@router.get(
        "/health",
        response_model=HealthResponse,
        summary="Работоспособность сервиса"
)
async def health():
    """Проверка работоспособности сервиса"""
    return await config_services.check_services()

@router.get(
        "/metrics",
        response_model=List[str],
        summary="Получить список доступных метрик"
)
async def get_available_metrics():
    """
    Возвращает список доступных метрик
    """
    try:
        return list(METRICS_REGISTRY.keys())
    except Exception as e:
        logger.error(f"Ошибка при получении списка метрик: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить список доступных метрик"
        )

@router.get(
        "/augmentations",
        response_model=List[str],
        summary="Получить список доступных методов аугментации"
)
async def get_available_augmentations():
    """
    Возвращает список доступных аугментаций
    """
    try:
        return list(ALLOWED_TRANSFORMS.keys())
    except Exception as e:
        logger.error(f"Ошибка при получении списка методов аугментации: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось получить список доступных методов аугментации"
        )
