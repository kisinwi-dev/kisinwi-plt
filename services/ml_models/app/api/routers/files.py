from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, Response

from app.logs import get_logger
from app.api.schemas import Files, File, FileDeletes
from app.api.deps import get_files_manager, FilesManager
from app.core.utils import valid_uuid

routers = APIRouter(
    prefix='/models/{model_id}/files',
    tags=['models', 'files']
)

logger = get_logger(__name__)

@routers.get(
    "",
    summary="Получение информации о файлах модели",
    responses={
        200: {"description": "Информация о файлах модели успешно получена"},
        204: {"description": "Модель не имеет файлов"},
        404: {"description": "Модель не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_files(
    model_id: str,
    manager: FilesManager = Depends(get_files_manager)
):
    try:
        valid_uuid(model_id, True)
        files = manager.get_info_files(model_id)
        if files:
            return Files(
                files=[
                    File(**file)
                    for file in files
                ]
            )
        else:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ValueError as e:
        logger.error(f"Получен не валидный model_id('{model_id}')")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )

@routers.post(
    "",
    summary="Загрузить файл модели",
    responses={
        200: {"description": "Файл успешно загружен"},
        404: {"description": "Модель не найдена"},
        409: {"description": "Ошибка сохранения файла"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def add_file(
    model_id: str,
    files: UploadFile,
    manager: FilesManager = Depends(get_files_manager)
):
    try:
        valid_uuid(model_id, True)
        manager.add_file(model_id, files)
    except ValueError as e:
        logger.error(f"Получен не валидный model_id('{model_id}')")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )
    except FileExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Файл уже существует"
        )

@routers.delete(
    "",
    summary="Удаление файлов модели",
    responses={
        200: {"description": "Операция удалёния выполнена"},
        204: {"description": "Нет файлов для удаления"},
        404: {"description": "Модель не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def del_file(
    model_id: str,
    files: FileDeletes,
    manager: FilesManager = Depends(get_files_manager)
):
    try:
        valid_uuid(model_id, True)

        if files.ids is None:
            result = manager.drop(model_id)
        else:
            result = manager.drop(model_id, files.ids)

        if result == 0:
            return Response(
                status_code=status.HTTP_204_NO_CONTENT,
                content="Нет файлов для удаления"
            )

    except ValueError as e:
        logger.error(f"Получен не валидный model_id('{model_id}')")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )
