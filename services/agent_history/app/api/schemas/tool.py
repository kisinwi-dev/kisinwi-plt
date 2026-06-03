from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ToolStatus(Enum):
    IN_PROGRESS = "IN PROGRESS"
    SUCCED = "SUCCEED"
    ERROR = "ERROR"

class Tool(BaseModel):
    id: str = Field(..., description="Id инструмента")
    agent_role: str = Field(..., description="Роль агента, который использует инструмент")
    name: str = Field(..., description="Название инструмента")
    status: ToolStatus = Field(..., description="Статус")
    message: str = Field(..., description="Сообщение от системы")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время события")
    input_args: Optional[dict[str, Any]] = Field(None, description="Входные аргументы инструмента")
    output: Optional[Any] = Field(None, description="Результат выполнения инструмента")
    duration_ms: Optional[float] = Field(None, description="Время выполнения в миллисекундах")
    error_traceback: Optional[str] = Field(None, description="Полный стектрейс ошибки")
