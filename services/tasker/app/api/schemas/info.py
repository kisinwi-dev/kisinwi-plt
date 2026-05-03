from typing import Dict, Any, List
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Схема для выдачи информации о работе сервиса"""
    status: str
    info: List[Dict[str, Any]] = Field(..., description="Список с информацией о соединениях")