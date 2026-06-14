from enum import Enum
from typing import Dict, List
from pydantic import BaseModel, Field

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthResponse(BaseModel):
    """Схема для выдачи информации о работе сервиса"""
    status: HealthStatus = Field(description="Статус сервиса")
    services: Dict[str, HealthStatus] = Field(description="Описание информации о состоянии требуемых сервисов")


class LlmModelInfo(BaseModel):
    """Описание одной модели из каталога для UI."""
    id: str = Field(description="OpenRouter id модели")
    label: str = Field(description="Человекочитаемое имя модели")
    supports_temperature: bool = Field(description="Принимает ли модель кастомный temperature")
    notes: str = Field(default="", description="Заметки и ограничения по модели")


class LlmSettingsResponse(BaseModel):
    """Текущее состояние настройки модели агентов."""
    current_model: str = Field(description="Сейчас выбранная модель агентов")
    default_model: str = Field(description="Модель по умолчанию (из env)")
    is_custom: bool = Field(description="Текущая модель вне каталога (кастомная)")
    available: List[LlmModelInfo] = Field(description="Каталог доступных моделей")


class SetLlmModelRequest(BaseModel):
    """Запрос на смену глобальной модели агентов."""
    model: str = Field(..., min_length=1, description="OpenRouter id модели")