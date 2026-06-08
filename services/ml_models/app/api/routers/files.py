from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, Response
from fastapi.responses import FileResponse

from app.logs import get_logger
from app.api.schemas import Files, File, FileDeletes
from app.api.deps import (
    get_files_manager,
    FilesManager,
    get_ml_models_manager,
    MlModelsManager,
    validate_model_id,
    validate_file_id,
)

routers = APIRouter(
    prefix='/models',
    tags=['models', 'files']
)

logger = get_logger(__name__)

@routers.get(
    "/{model_id}/files",
    summary="Получение информации о файлах модели",
    description="Возвращает список файлов, связанных с указанной моделью",
    response_description="Список файлов модели",
    responses={
        200: {"description": "Информация о файлах модели успешно получена"},
        204: {"description": "Модель не имеет файлов"},
        404: {"description": "Модель не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_files(
    model_id: str = Depends(validate_model_id),
    manager: FilesManager = Depends(get_files_manager),
    models_manager: MlModelsManager = Depends(get_ml_models_manager),
):
    # Модель должна существовать — иначе 404 (а не «нет файлов»)
    if models_manager.get_model(model_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    files = manager.get_info_files(model_id)
    if files:
        return Files(files=[File(**file) for file in files])

    return Response(status_code=status.HTTP_204_NO_CONTENT)

@routers.post(
    "/{model_id}/files",
    summary="Загрузить файл модели",
    description="Загружает и сохраняет файл, связанный с указанной моделью",
    response_description="Пустой ответ при успешной загрузке",
    responses={
        200: {"description": "Файл успешно загружен"},
        404: {"description": "Модель не найдена"},
        409: {"description": "Ошибка сохранения файла"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def add_file(
    files: UploadFile,
    model_id: str = Depends(validate_model_id),
    manager: FilesManager = Depends(get_files_manager),
    models_manager: MlModelsManager = Depends(get_ml_models_manager)
):
    # Модель должна существовать — иначе 404 (а не 500 + директория-мусор)
    if models_manager.get_model(model_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    try:
        manager.add_file(model_id, files)
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

    return {"model_id": model_id, "filename": files.filename, "status": "ok"}

@routers.delete(
    "/{model_id}/files",
    summary="Удаление файлов модели",
    description="Удаляет указанные файлы модели, либо все файлы модели, если идентификаторы не переданы",
    response_description="Пустой ответ при успешном удалении",
    responses={
        200: {"description": "Операция удалёния выполнена"},
        204: {"description": "Нет файлов для удаления"},
        404: {"description": "Модель не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def del_file(
    files: FileDeletes,
    model_id: str = Depends(validate_model_id),
    manager: FilesManager = Depends(get_files_manager),
    models_manager: MlModelsManager = Depends(get_ml_models_manager)
):
    # Модель должна существовать — иначе 404 (а не 204 + директория-мусор)
    if models_manager.get_model(model_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

    if files.ids is None:
        result = manager.drop(model_id)
    else:
        result = manager.drop(model_id, files.ids)

    if result == 0:
        return Response(
            status_code=status.HTTP_204_NO_CONTENT,
            content="Нет файлов для удаления"
        )

@routers.get(
    "/files/{file_id}/download",
    summary="Скачать конкретный файл по ID",
    description="Возвращает содержимое файла модели для скачивания по его идентификатору",
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