from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Response, Query

from app.api.schemas import *
from app.api.deps import (
    get_models_manager,
    ModelsManager,
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
    summary="Получить модели с вложенными версиями",
    description="Возвращает модели с их версиями (версии по убыванию). Пагинация по моделям.",
    response_description="Список моделей с версиями",
    response_model=Models,
    responses={
        200: {"description": "Список моделей (возможно пустой)"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_models(
    dataset_id: Optional[str] = Query(None, description="Фильтр по ID датасета (по версиям)"),
    model_status: Optional[str] = Query(None, alias="status", description="Фильтр по статусу версии"),
    name: Optional[str] = Query(None, description="Фильтр по имени модели (частичное совпадение)"),
    limit: Optional[int] = Query(None, ge=1, description="Размер страницы (по моделям)"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    manager: ModelsManager = Depends(get_models_manager)
):
    """
    Модели с вложенными версиями.

    Фильтры status/dataset_id отбирают модели, у которых есть хотя бы одна
    подходящая версия. Пагинация необязательна: без `limit` возвращаются все.
    """
    result = manager.get_models(
        name=name,
        status=model_status,
        dataset_id=dataset_id,
        limit=limit,
        offset=offset,
    )
    return Models(**result)

@routers.post(
    "",
    summary="Создать модель",
    description="Создаёт новую модель (родительскую сущность с уникальным именем)",
    response_description="Идентификатор созданной модели",
    status_code=201,
    responses={
        201: {"description": "Модель успешно создана"},
        409: {"description": "Модель с таким именем уже существует"},
        422: {"description": "Ошибка валидации данных"}
    }
)
async def create_model(
    model: ModelCreate,
    manager: ModelsManager = Depends(get_models_manager)
):
    try:
        model_id = manager.create(**model.model_dump())
        return {"model_id": model_id}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ошибка в запросе: {e}"
        )

@routers.get(
    "/statistics",
    summary="Получить статистику по моделям",
    description="Возвращает количество моделей, версий и распределение версий по статусам",
    response_description="Счётчики моделей и версий",
    response_model=MLModelsStatistics,
    responses={
        200: {"description": "Статистика успешно собрана"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_models_statistics(
    manager: ModelsManager = Depends(get_models_manager)
):
    """Статистика: моделей всего, версий всего и по статусам (для дашборда)"""
    return manager.get_statistics()

@routers.get(
    "/by-name/{name}",
    summary="Получить модель по имени",
    description="Возвращает модель с версиями по точному имени",
    response_description="Модель с версиями",
    response_model=Model,
    responses={
        200: {"description": "Модель найдена"},
        404: {"description": "Модель не найдена"}
    }
)
async def get_model_by_name(
    name: str,
    manager: ModelsManager = Depends(get_models_manager)
):
    model = manager.get_by_name(name)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с именем '{name}' не найдена"
        )

    return Model(**model)

@routers.get(
    "/{model_id}",
    summary="Получить информацию о модели",
    description="Возвращает модель с её версиями по идентификатору",
    response_description="Модель с версиями",
    response_model=Model,
    responses={
        200: {"description": "Модель успешно найдена"},
        404: {"description": "Модель не найдена"}
    }
)
async def get_model_by_id(
    model_id: str = Depends(validate_model_id),
    manager: ModelsManager = Depends(get_models_manager)
):
    model = manager.get_by_id(model_id)

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    return Model(**model)

@routers.patch(
    "/{model_id}",
    summary="Частичное обновление модели",
    description="Обновляет имя и/или описание модели",
    response_description="Пустой ответ при успешном обновлении",
    responses={
        200: {"description": "Модель успешно обновлена"},
        400: {"description": "Некорректные данные запроса"},
        404: {"description": "Модель не найдена"},
        409: {"description": "Модель с таким именем уже существует"}
    }
)
async def update_model(
    update_data: ModelUpdate,
    model_id: str = Depends(validate_model_id),
    manager: ModelsManager = Depends(get_models_manager)
):
    update_dict = update_data.model_dump(exclude_none=True)

    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нет данных для обновления"
        )

    try:
        updated = manager.update(model_id, update_dict)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    return Response(status_code=200)

@routers.delete(
    "/{model_id}",
    summary="Удалить модель",
    description="Удаляет модель со всеми версиями и связанными файлами",
    response_description="Пустой ответ при успешном удалении",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Модель успешно удалена"},
        404: {"description": "Модель не найдена"}
    }
)
async def delete_model(
    model_id: str = Depends(validate_model_id),
    manager: ModelsManager = Depends(get_models_manager),
    files_manager: FilesManager = Depends(get_files_manager)
):
    deleted_version_ids = manager.delete(model_id)

    if deleted_version_ids is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    # Записи удалены (FK CASCADE убрал версии и метаданные файлов) — чистим диск
    for version_id in deleted_version_ids:
        files_manager.drop_version_dir(version_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
