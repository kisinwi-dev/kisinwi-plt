from typing import Dict, Any
from fastapi import APIRouter, Body, HTTPException, status

from app.logs import get_logger
from app.core.utils import validate_task_params
from app.api.schemas import TaskParams, ConfigValidationResponse

logger = get_logger(__name__)

routers = APIRouter(
    prefix='/config',
    tags=['Validation']
)

@routers.post(
        "/validate",
        response_model=ConfigValidationResponse,
        summary="Проверить конфигурацию обучения"
)
async def validate_config(
    config: Dict[str, Any] = Body(
        ...,
        description="Конфигурация обучения (структура — GET /info/example_config)",
        examples=[TaskParams.model_config["json_schema_extra"]["example"]]
    )
) -> ConfigValidationResponse:
    """
    Проверяет конфигурацию обучения без запуска: соответствие схеме,
    существование модели, функции потерь, оптимизатора, планировщика,
    метрик, трансформаций и доступность устройства на этом инстансе.
    """
    try:
        errors = validate_task_params(config)
        return ConfigValidationResponse(valid=not errors, errors=errors)
    except Exception as e:
        logger.error(f"Ошибка при валидации конфигурации: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не удалось выполнить валидацию конфигурации"
        )
