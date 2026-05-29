from typing import Dict, Any
from pydantic import BaseModel

class HealthResponse(BaseModel):
    """Схема для выдачи информации о работе сервиса"""
    status: str
    info: Dict[str, Any]