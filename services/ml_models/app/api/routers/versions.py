from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Response, Query

from app.api.schemas import *
from app.api.deps import (
    get_versions_manager,
    VersionsManager,
    get_files_manager,
    FilesManager,
    validate_model_id,
    validate_version_id,
)

routers = APIRouter(
    tags=['versions']
)

@routers.post(
    "/models/{model_id}/versions",
    summary="Создать версию модели",
    description="Создаёт новую версию модели; номер версии назначается сервером (MAX+1)",
    response_description="Идентификатор и номер созданной версии",
    status_code=201,
    responses={
        201: {"description": "Версия успешно создана"},
        404: {"description": "Модель не найдена"},
        422: {"description": "Ошибка валидации данных"}
    }
)
async def create_version(
    version: ModelVersionCreate,
    model_id: str = Depends(validate_model_id),
    manager: VersionsManager = Depends(get_versions_manager)
):
    try:
        version_id, version_number = manager.create(
            model_id=model_id,
            **version.model_dump()
        )
        return {"version_id": version_id, "version": version_number}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )

@routers.get(
    "/versions",
    summary="Получить плоский список версий",
    description="Возвращает версии всех моделей (свежие сверху) с опциональной фильтрацией",
    response_description="Список версий",
    response_model=ModelVersions,
    responses={
        200: {"description": "Список версий (возможно пустой)"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_versions(
    dataset_id: Optional[str] = Query(None, description="Фильтр по ID датасета"),
    version_status: Optional[str] = Query(None, alias="status", description="Фильтр по статусу версии"),
    name: Optional[str] = Query(None, description="Фильтр по имени модели (частичное совпадение)"),
    model_id: Optional[str] = Query(None, description="Фильтр по ID родительской модели"),
    limit: Optional[int] = Query(None, ge=1, description="Размер страницы (без параметра — все версии)"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    manager: VersionsManager = Depends(get_versions_manager)
):
    """
    Плоский список версий (свежие сверху).

    Пагинация необязательна: без `limit` возвращаются все версии. Всегда
    возвращает 200 с JSON-списком и общим количеством `total` (с учётом фильтров).
    """
    versions = manager.get_versions(
        name=name,
        status=version_status,
        dataset_id=dataset_id,
        model_id=model_id,
        limit=limit,
        offset=offset,
    )
    total = manager.count_versions(
        name=name,
        status=version_status,
        dataset_id=dataset_id,
        model_id=model_id,
    )

    items = [ModelVersion(**version) for version in versions]

    return ModelVersions(versions=items, total=total, limit=limit, offset=offset)

@routers.get(
    "/versions/{version_id}",
    summary="Получить информацию о версии",
    description="Возвращает полную информацию о версии (с именем и описанием модели)",
    response_description="Метаданные версии",
    response_model=ModelVersion,
    responses={
        200: {"description": "Версия успешно найдена"},
        404: {"description": "Версия не найдена"}
    }
)
async def get_version_by_id(
    version_id: str = Depends(validate_version_id),
    manager: VersionsManager = Depends(get_versions_manager)
):
    version = manager.get_version(version_id)

    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Версия с ID {version_id} не найдена"
        )

    return ModelVersion(**version)

@routers.patch(
    "/versions/{version_id}",
    summary="Частичное обновление версии",
    description="Обновляет переданные поля указанной версии; неуказанные поля остаются прежними",
    response_description="Пустой ответ при успешном обновлении",
    responses={
        200: {"description": "Версия успешно обновлена"},
        400: {"description": "Некорректные данные запроса"},
        404: {"description": "Версия не найдена"}
    }
)
async def update_version(
    update_data: ModelVersionUpdate,
    version_id: str = Depends(validate_version_id),
    manager: VersionsManager = Depends(get_versions_manager)
):
    """
    Частичное обновление версии.
    Можно обновить любое из полей, передав только нужные.
    """
    update_dict = update_data.model_dump(exclude_none=True)

    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нет данных для обновления"
        )

    # Один запрос: UPDATE ... RETURNING вернёт False, если версии нет → 404
    try:
        updated = manager.update_version(version_id, update_dict)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Версия с ID {version_id} не найдена"
        )

    return Response(status_code=200)

@routers.delete(
    "/versions/{version_id}",
    summary="Удалить версию",
    description="Удаляет указанную версию вместе с её файлами",
    response_description="Пустой ответ при успешном удалении",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Версия успешно удалена"},
        404: {"description": "Версия не найдена"}
    }
)
async def delete_version(
    version_id: str = Depends(validate_version_id),
    manager: VersionsManager = Depends(get_versions_manager),
    files_manager: FilesManager = Depends(get_files_manager)
):
    deleted = manager.delete_version(version_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Версия с ID {version_id} не найдена"
        )

    # Запись версии удалена (FK CASCADE убрал метаданные файлов) — чистим диск
    files_manager.drop_version_dir(version_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
