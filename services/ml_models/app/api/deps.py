from fastapi import HTTPException, status

from app.core.train_models_tasks import MlModelsManager
from app.core.files import FilesManager
from app.core.utils import valid_uuid

ml_models_manager = MlModelsManager()
files_manager = FilesManager()

async def get_ml_models_manager() -> MlModelsManager:
    return ml_models_manager

async def get_files_manager() -> FilesManager:
    return files_manager

def validate_model_id(model_id: str) -> str:
    """Зависимость: валидирует UUID модели из пути, иначе 404."""
    if not valid_uuid(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Модель с ID {model_id} не найдена"
        )
    return model_id

def validate_file_id(file_id: str) -> str:
    """Зависимость: валидирует UUID файла из пути, иначе 404."""
    if not valid_uuid(file_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Файл с ID {file_id} не найден"
        )
    return file_id