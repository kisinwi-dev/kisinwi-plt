from typing import List, Tuple

from .base import CoreException

class DatasetAlreadyExistsError(CoreException):
    """Датасет с таким идентификатором уже существует"""
    
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Датасет с идентификатором '{dataset_id}' уже существует",
            status_code=409,
            detail="Попробуйте использовать другой идентификатор."
        )


class DatasetMetadataFileError(CoreException):
    """Не найден файл с метаданными"""
    
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"В датасете с идентификатором '{dataset_id}' остуствует файл с метаданными",
            status_code=409,
        )

class CannotDeleteDefaultVersion(CoreException):
    """Удалени дефолтной версии"""
    
    def __init__(self, dataset_id: str, version_id: str):
        super().__init__(
            message=(
                "Нельзя удалять стандартные версии"
                f"В датасете '{dataset_id}' версия {version_id} является стандартной версией."
            ),
            status_code=409,
        )

class DatasetNotFoundError(CoreException):
    """Датасет не найден"""
    
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Датасет с идентификатором '{dataset_id}' не найден",
            status_code=404,
            detail="Убедитесь, что идентификатор указан верно."
        )


class DatasetValidationError(CoreException):
    """Структура или содержимое датасета не соответствует требованиям"""
    
    def __init__(self, reason: str, dataset_id: str | None = None):
        msg = f"Ошибка валидации датасета"
        if dataset_id:
            msg += f" '{dataset_id}'"
        msg += f": {reason}"
        
        super().__init__(
            message=msg,
            status_code=400,
            detail=reason
        )

class UnsupportedDatasetError(CoreException):
    """Задача и тип датасета не поддерживаются"""
    
    def __init__(
        self,
        dataset_task: str,
        dataset_type: str | None = None,
        supported: List[Tuple[str, str]] | None = None
    ):
        msg = f"Задача '{dataset_task}' не поддерживается"
        if dataset_type:
            msg += f" для типа '{dataset_type}'"
        if supported:
            msg += f" (доступные вариации: {supported})"
        
        super().__init__(
            message=msg,
            status_code=400,
            detail=f"Поддерживаемые задачи для типа '{dataset_type}': {supported if supported else ' _ERROR_ '}"
        )