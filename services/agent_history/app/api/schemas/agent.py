from datetime import datetime
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
