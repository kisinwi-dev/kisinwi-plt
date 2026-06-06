from typing import Optional

from psycopg2 import OperationalError, InterfaceError
from fastapi import APIRouter, Depends, HTTPException, status, Response, Query

from app.logs import get_logger
from app.api.schemas import *
from app.api.deps import get_ml_models_manager, MlModelsManager
from app.core.utils import valid_uuid

routers = APIRouter(
    prefix='/models',
    tags=['models']
)

logger = get_logger(__name__)

@routers.get(
    "",
    summary="Получить информацию о всех моделях",
    response_model=MLModels,
    responses={
        200: {"description": "Список моделей (возможно пустой)"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_models(
    dataset_id: Optional[str] = Query(None, description="Фильтр по ID датасета"),
    model_status: Optional[str] = Query(None, alias="status", description="Фильтр по статусу модели"),
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    """
    Получить полную информацию о моделях (свежие сверху).

    Опционально фильтрует по датасету и/или статусу. Всегда возвращает 200 с
    JSON-списком; при отсутствии моделей список пустой.
    """

    # Получение моделей
    models = manager.get_model(dataset_id=dataset_id, status=model_status)

    if models is None:
        return MLModels(models=[])

    return MLModels(models=[MLModel(**model) for model in models])

@routers.post(
    "",
    summary="Создание модели",
    status_code=201,
    responses={
        201: {"description": "Модель успешно создана"},
        400: {"description": "Некорректные данные"},
        409: {"description": "Модель с таким name и version уже существует"},
        422: {"description": "Ошибка валидации данных"}
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

@routers.delete(
    "/{model_id}",
    summary="Удаление модели",
    status_code=status.HTTP_200_OK,
    responses={
        204: {"description": "Модель успешно удалена"},
        404: {"description": "Модель не найдена"}
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
                
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Некорректный формат введённых данных: {e}"
        )

@routers.get(
    "/{model_id}",
    summary="Получить информацию о модели",
    response_model=MLModel,
    responses={
        200: {"description": "Модель успешно найдена"},
        404: {"description": "Модель не найдена"}
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
    models = manager.get_model(model_id)

    if models is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    model = MLModel(**models[0])

    return model

@routers.patch(
    "/{model_id}",
    summary="Частичное обновление модели",
    responses={
        200: {"description": "Модель успешно обновлена"},
        400: {"description": "Некорректные данные запроса"},
        404: {"description": "Модель не найдена"}
    }
)
async def update_model(
    model_id: str,
    update_data: MLModelUpdate,
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    """
    Частичное обновление модели.
    Можно обновить любое из полей, передав только нужные.
    """

    try:
        valid_uuid(model_id, True)
    except ValueError as e:
        logger.error(f"Получен не валидный model_id = '{model_id}'")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    # Получаем словарь без None значений
    update_dict = update_data.model_dump(exclude_none=True)

    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нет данных для обновления"
        )

    # Проверка существования модели до обновления
    if manager.get_model(model_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    try:
        # Выполняем обновление
        updated_model = manager.update_model(model_id, update_dict)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    if not updated_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    return Response(status_code=200)