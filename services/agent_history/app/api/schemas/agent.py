from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class AgentStatus(Enum):
    IN_PROGRESS = "IN PROGRESS"
    SUCCED = "SUCCEED"
    ERROR = "ERROR"


class AgentResponse(BaseModel):
    """Схема ответа агента"""
    response_id: str = Field(..., description="Уникальный ID ответа")
    status: AgentStatus = Field(default=AgentStatus.IN_PROGRESS, description="Тип")
    agent_role: str = Field(..., description="Роль агента")
    text: str = Field(..., description="Текст ответа агента", min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")
    model: Optional[str] = Field(None, description="Модель LLM (gpt-4o, claude-3-5-sonnet…)")
    duration_ms: Optional[float] = Field(None, description="Время выполнения агента, мс")
    task_name: Optional[str] = Field(None, description="Название задачи агента")
    iteration: Optional[int] = Field(None, description="Номер итерации")
