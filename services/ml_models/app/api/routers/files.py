from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, Response
from fastapi.responses import FileResponse

from app.logs import get_logger
from app.api.schemas import Files, File, FileDeletes
from app.api.deps import (
    get_files_manager,
    FilesManager,
    get_versions_manager,
    VersionsManager,
    validate_version_id,
    validate_file_id,
)

routers = APIRouter(
    tags=['versions', 'files']
)

logger = get_logger(__name__)

@routers.get(
    "/versions/{version_id}/files",
    summary="Получение информации о файлах версии",
    description="Возвращает список файлов, связанных с указанной версией модели",
    response_description="Список файлов версии",
    responses={
        200: {"description": "Информация о файлах версии успешно получена"},
        204: {"description": "Версия не имеет файлов"},
        404: {"description": "Версия не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_files(
    version_id: str = Depends(validate_version_id),
    manager: FilesManager = Depends(get_files_manager),
    versions_manager: VersionsManager = Depends(get_versions_manager),
):
    # Версия должна существовать — иначе 404 (а не «нет файлов»)
    if versions_manager.get_version(version_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Версия с ID {version_id} не найдена"
        )

    files = manager.get_info_files(version_id)
    if files:
        return Files(files=[File(**file) for file in files])

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@routers.post(
    "/versions/{version_id}/files",
    summary="Загрузить файл версии",
    description="Загружает и сохраняет файл, связанный с указанной версией модели",
    response_description="Пустой ответ при успешной загрузке",
    responses={
        200: {"description": "Файл успешно загружен"},
        404: {"description": "Версия не найдена"},
        409: {"description": "Ошибка сохранения файла"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def add_file(
    files: UploadFile,
    version_id: str = Depends(validate_version_id),
    manager: FilesManager = Depends(get_files_manager),
    versions_manager: VersionsManager = Depends(get_versions_manager)
):
    # Версия должна существовать — иначе 404 (а не 500 + директория-мусор)
    if versions_manager.get_version(version_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Версия с ID {version_id} не найдена"
        )

    try:
        manager.add_file(version_id, files)
    except FileExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Файл уже существует"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    return {"version_id": version_id, "filename": files.filename, "status": "ok"}

@routers.delete(
    "/versions/{version_id}/files",
    summary="Удаление файлов версии",
    description="Удаляет указанные файлы версии, либо все файлы версии, если идентификаторы не переданы",
    response_description="Пустой ответ при успешном удалении",
    responses={
        200: {"description": "Операция удалёния выполнена"},
        204: {"description": "Нет файлов для удаления"},
        404: {"description": "Версия не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def del_file(
    files: FileDeletes,
    version_id: str = Depends(validate_version_id),
    manager: FilesManager = Depends(get_files_manager),
    versions_manager: VersionsManager = Depends(get_versions_manager)
):
    # Версия должна существовать — иначе 404 (а не 204 + директория-мусор)
    if versions_manager.get_version(version_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Версия с ID {version_id} не найдена"
        )

    if files.ids is None:
        result = manager.drop(version_id)
    else:
        result = manager.drop(version_id, files.ids)

    if result == 0:
        return Response(
            status_code=status.HTTP_204_NO_CONTENT,
            content="Нет файлов для удаления"
        )

@routers.get(
    "/files/{file_id}/download",
    summary="Скачать конкретный файл по ID",
    description="Возвращает содержимое файла версии для скачивания по его идентификатору",
    response_description="Содержимое файла",
    responses={
        200: {"description": "Файл успешно скачан"},
        404: {"description": "Файл не найден"},
        400: {"description": "Невалидный UUID"}
    }
)
async def download_file(
    file_id: str = Depends(validate_file_id),
    manager: FilesManager = Depends(get_files_manager)
):
    """
    Скачать конкретный файл по его ID.
    """
    try:
        file_path, filename = manager.get_file_path(file_id)

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Ошибка скачивания файла: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка скачивания файла"
        )
