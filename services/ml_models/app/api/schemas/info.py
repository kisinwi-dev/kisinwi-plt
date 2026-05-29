from typing import Dict, Any, List
from pydantic import BaseModel, Field

from .ml_models import MLModelStatus
from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthResponse(BaseModel):
    """Схема для выдачи информации о работе сервиса"""
    status: HealthStatus = Field(description="Статус сервиса")
    services: Dict[str, HealthStatus] = Field(description="Описание информации о состоянии требуемых сервисов")

class MLModelsStatusesResponse(BaseModel):
    """Ответ со списком статусов"""
    statuses: List[MLModelStatus] = Field(..., description="Список возможных статусов модели и их описание")