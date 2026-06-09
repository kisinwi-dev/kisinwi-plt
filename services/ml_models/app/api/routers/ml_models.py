from typing import Optional

from psycopg2 import OperationalError, InterfaceError
from fastapi import APIRouter, Depends, HTTPException, status, Response, Query

from app.api.schemas import *
from app.api.deps import (
    get_ml_models_manager,
    MlModelsManager,
    get_files_manager,
    FilesManager,
    validate_model_id,
)

routers = APIRouter(
    prefix='/models',
    tags=['models']
)

@routers.get(
    "",
    summary="Получить информацию о всех моделях",
    description="Возвращает список моделей (свежие сверху) с опциональной фильтрацией по датасету и статусу",
    response_description="Список метаданных моделей",
    response_model=MLModels,
    responses={
        200: {"description": "Список моделей (возможно пустой)"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_models(
    dataset_id: Optional[str] = Query(None, description="Фильтр по ID датасета"),
    model_status: Optional[str] = Query(None, alias="status", description="Фильтр по статусу модели"),
    name: Optional[str] = Query(None, description="Фильтр по имени модели (все версии модели)"),
    limit: Optional[int] = Query(None, ge=1, description="Размер страницы (без параметра — все модели)"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    """
    Получить полную информацию о моделях (свежие сверху).

    Опционально фильтрует по датасету, статусу и/или имени (все версии модели).
    Пагинация необязательна: без `limit` возвращаются все модели. Всегда
    возвращает 200 с JSON-списком и общим количеством `total` (с учётом фильтров).
    """

    # Получение моделей и общего количества с учётом фильтров
    models = manager.get_model(
        dataset_id=dataset_id,
        status=model_status,
        name=name,
        limit=limit,
        offset=offset
    )
    total = manager.count_model(dataset_id=dataset_id, status=model_status, name=name)

    items = [MLModel(**model) for model in models] if models else []

    return MLModels(models=items, total=total, limit=limit, offset=offset)

@routers.post(
    "",
    summary="Создать модель",
    description="Создаёт новую ML модель с её параметрами обучения и привязкой к датасету",
    response_description="Идентификатор созданной модели",
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

@routers.get(
    "/statistics",
    summary="Получить статистику по моделям",
    description="Возвращает общее количество моделей и распределение по статусам",
    response_description="Общее количество и счётчики моделей по статусам",
    response_model=MLModelsStatistics,
    responses={
        200: {"description": "Статистика успешно собрана"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_models_statistics(
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    """Статистика моделей: всего и по статусам (для дашборда)"""
    return manager.get_statistics()

@routers.get(
    "/grouped",
    summary="Получить модели, сгруппированные по имени",
    description="Возвращает модели сгруппированными по имени, версии отсортированы по убыванию. Пагинация по уникальным именам.",
    response_description="Список групп моделей",
    response_model=MLModelsGrouped,
    responses={
        200: {"description": "Список групп (возможно пустой)"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_grouped_models(
    dataset_id: Optional[str] = Query(None, description="Фильтр по ID датасета"),
    model_status: Optional[str] = Query(None, alias="status", description="Фильтр по статусу модели"),
    name: Optional[str] = Query(None, description="Фильтр по имени модели (частичное совпадение)"),
    limit: Optional[int] = Query(None, ge=1, description="Размер страницы (по именам)"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    result = manager.get_grouped_models(
        dataset_id=dataset_id,
        status=model_status,
        name=name,
        limit=limit,
        offset=offset,
    )
    return MLModelsGrouped(**result)

@routers.delete(
    "/by-name/{name}",
    summary="Удалить все версии модели по имени",
    status_code=status.HTTP_200_OK,
)
async def delete_models_by_name(
    name: str,
    manager: MlModelsManager = Depends(get_ml_models_manager),
    files_manager: FilesManager = Depends(get_files_manager),
):
    deleted_ids = manager.delete_by_name(name)
    if not deleted_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модели с именем '{name}' не найдены",
        )

    # Записи удалены (FK CASCADE убрал метаданные файлов) — чистим диск по каждой версии
    for model_id in deleted_ids:
        files_manager.drop_model_dir(model_id)

    return {"deleted": len(deleted_ids)}


@routers.delete(
    "/{model_id}",
    summary="Удалить модель",
    description="Удаляет указанную модель по её идентификатору вместе со связанными данными",
    response_description="Пустой ответ при успешном удалении",
    status_code=status.HTTP_200_OK,
    responses={
        204: {"description": "Модель успешно удалена"},
        404: {"description": "Модель не найдена"}
    }
)
async def delete_task(
    model_id: str = Depends(validate_model_id),
    manager: MlModelsManager = Depends(get_ml_models_manager),
    files_manager: FilesManager = Depends(get_files_manager)
):
    deleted = manager.delete(model_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    # Запись модели удалена (FK CASCADE убрал метаданные файлов) — чистим диск
    files_manager.drop_model_dir(model_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@routers.get(
    "/{model_id}",
    summary="Получить информацию о модели",
    description="Возвращает полную информацию об указанной модели по её идентификатору",
    response_description="Метаданные модели",
    response_model=MLModel,
    responses={
        200: {"description": "Модель успешно найдена"},
        404: {"description": "Модель не найдена"}
    }
)
async def get_model_by_id(
    model_id: str = Depends(validate_model_id),
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    """Получить полную информацию о модели по её ID"""

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
    description="Обновляет переданные поля указанной модели; неуказанные поля остаются прежними",
    response_description="Пустой ответ при успешном обновлении",
    responses={
        200: {"description": "Модель успешно обновлена"},
        400: {"description": "Некорректные данные запроса"},
        404: {"description": "Модель не найдена"}
    }
)
async def update_model(
    update_data: MLModelUpdate,
    model_id: str = Depends(validate_model_id),
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    """
    Частичное обновление модели.
    Можно обновить любое из полей, передав только нужные.
    """

    # Получаем словарь без None значений
    update_dict = update_data.model_dump(exclude_none=True)

    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нет данных для обновления"
        )

    # Один запрос: UPDATE ... RETURNING вернёт False, если модели нет → 404
    try:
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