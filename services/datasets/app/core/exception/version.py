from .base import CoreException

class VersionNotFoundError(CoreException):
    """Версия не найден"""
    
    def __init__(self, version_id: str):
        super().__init__(
            message=f"Версия с идентификатором '{version_id}' не найдена",
            status_code=404,
            detail="Убедитесь, что идентификатор указан верно."
        )

class VersionValidationError(CoreException):
    """Структура или содержимое версии не соответствует требованиям"""
    
    def __init__(
            self, 
            reason: str, 
            version_id: str | None = None
    ):
        msg = f"Ошибка валидации датасета: "
        
        if version_id:
            msg += f" '{version_id}'."    
        
        super().__init__(
            message=msg,
            status_code=400,
            detail=reason
        )