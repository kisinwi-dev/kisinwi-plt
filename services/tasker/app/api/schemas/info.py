from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthResponse(BaseModel):
    """Схема для выдачи информации о работе сервиса"""
    status: HealthStatus = Field(description="Статус сервисаё")
    services: Dict[str, HealthStatus] = Field(description="Описание информации о состоянии требуемых сервисов")