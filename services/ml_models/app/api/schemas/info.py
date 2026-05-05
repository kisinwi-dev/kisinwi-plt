from typing import Dict, Any, List
from pydantic import BaseModel, Field

from .ml_models import MLModelStatusInfo

class HealthResponse(BaseModel):
    """Схема для выдачи информации о работе сервиса"""
    status: str
    info: List[Dict[str, Any]] = Field(..., description="Список с информацией о соединениях")

class MLModelsStatusesResponse(BaseModel):
    """Ответ со списком статусов"""
    statuses: List[MLModelStatusInfo] = Field(..., description="Список возможных статусов модели и их описание")