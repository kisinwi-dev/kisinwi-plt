from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class SystemMessageType(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    AGENT_START = "AGENT_START"
    AGENT_RESPONSE = "AGENT_RESPONSE"

class SystemMessage(BaseModel):
    """Модель для создания сообщения от системы"""
    type_: SystemMessageType = Field(default_factory=SystemMessageType, description="Тип сообщения")
    message: str = Field(..., description="Сообщение от системы")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время события")

    def model_dump_json_custom(self) -> str:
        """Кастомная сериализация с русскими символами"""
        return self.model_dump_json(indent=2)

class AgentResponse(BaseModel):
    """Модель ответа агента"""
    type_: SystemMessageType = Field(default=SystemMessageType.AGENT_RESPONSE, description="Тип")

    response_id: str = Field(..., description="Уникальный ID ответа")
    agent_role: str = Field(..., description="Роль агента")
    text: str = Field(..., description="Текст ответа агента", min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")

    def model_dump_json_custom(self) -> str:
        """Кастомная сериализация с русскими символами"""
        return self.model_dump_json(indent=2)