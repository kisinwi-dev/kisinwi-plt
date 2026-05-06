from fastapi import APIRouter, Depends, HTTPException, status, UploadFile

from app.logs import get_logger
from app.api.schemas import *
from app.api.deps import get_files_manager, FilesManager
from app.core.utils import valid_uuid

routers = APIRouter(
    prefix='/models/{model_id}',
    tags=['models', 'files']
)

logger = get_logger(__name__)

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
async def get_models(
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
