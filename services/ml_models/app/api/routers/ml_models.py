from psycopg2 import OperationalError, InterfaceError
from fastapi import APIRouter, Depends, HTTPException, status, Response

from app.logs import get_logger
from app.api.schemas import *
from app.api.deps import get_ml_models_manager, MlModelsManager
from app.core.utils import valid_uuid

routers = APIRouter(
    prefix='/models',
    tags=['models']
)

logger = get_logger(__name__)

@routers.post(
    "",
    summary="Создание модели",
    status_code=201,
    responses={
        201: {"description": "Модель успешно создана"},
        400: {"description": "Некорректные данные"},
        409: {"description": "Модель с таким name и version уже существует"},
        422: {"description": "Ошибка валидации данных"},
        500: {"description": "Внутренняя ошибка сервера"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def create_model(
    model: MLModelCreate,
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    try:
        model_id = manager.create(
            **model.model_dump()
        )
        return {"model_id": model_id}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ошибка в запросе: {e}"
        )
    except (OperationalError, InterfaceError) as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Ошибка подключения к БД: {e}"
        )

@routers.delete(
    "/{model_id}",
    summary="Удаление модели",
    status_code=status.HTTP_200_OK,
    responses={
        204: {"description": "Модель успешно удалена"},
        404: {"description": "Модель не найдена"},
        503: {"description": "Ошибка подключения к БД"},
        500: {"description": "Внутренняя ошибка сервера"}
    }
)
async def delete_task(
    model_id: str,
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    try:
        deleted = manager.delete(model_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Модель с ID {model_id} не найдена"
            )
                
        return Response(
            status_code=204
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Некорректный формат введённых данных: {e}"
        )
    except (OperationalError, InterfaceError) as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Ошибка подключения к БД: {e}"
        )

@routers.get(
    "/{model_id}",
    summary="Получить информацию о модели",
    response_model=MLModel,
    responses={
        200: {"description": "Модель успешно найдена"},
        404: {"description": "Модель не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_model_by_id(
    model_id: str,
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    """Получить полную информацию о модели по её ID"""
    
    try:
        valid_uuid(model_id, True)
    except ValueError as e:
        logger.error(f"Получен не валидный model_id('{model_id}')")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    # Получение модели
    model = manager.get_model(model_id)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    model = MLModel(**model)

    return model
