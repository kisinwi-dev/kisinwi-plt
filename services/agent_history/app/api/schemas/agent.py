from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class AgentMessageType(Enum):
    START = "START"
    RESPONSE = "RESPONSE"
    
class AgentStart(BaseModel):
    """Схема старта работы агента"""
    type_: AgentMessageType = Field(default=AgentMessageType.START, description="Тип")
    response_id: str = Field(..., description="Уникальный ID ответа")
    agent_role: str = Field(..., description="Роль агента")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")

class AgentResponse(BaseModel):
    """Схема ответа агента"""
    type_: AgentMessageType = Field(default=AgentMessageType.RESPONSE, description="Тип")

    response_id: str = Field(..., description="Уникальный ID ответа")
    agent_role: str = Field(..., description="Роль агента")
    text: str = Field(..., description="Текст ответа агента", min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")
