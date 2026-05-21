from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class SystemMessageType(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    AGENT_ACTION = "AGENT_ACTION"

class SystemMessage(BaseModel):
    type_: SystemMessageType = Field(default_factory=SystemMessageType, description="Тип информации")
    desc: str = Field(..., description="Описание лога")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")

    def model_dump_json_custom(self) -> str:
        """Кастомная сериализация с русскими символами"""
        return self.model_dump_json(indent=2)

class AgentResponseCreate(BaseModel):
    """Модель создания ответа агента"""
    
    agent_role: str = Field(..., description="Роль агента")
    text: str = Field(..., description="Текст ответа агента", min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")
    
    def model_dump_json_custom(self) -> str:
        """Кастомная сериализация с русскими символами"""
        return self.model_dump_json(indent=2)

class AgentResponse(BaseModel):
    """Модель ответа агента"""
    
    agent_role: str = Field(..., description="Роль агента")
    text: str = Field(..., description="Текст ответа агента", min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")
    response_id: str = Field(..., description="Уникальный ID ответа")
    
    def model_dump_json_custom(self) -> str:
        """Кастомная сериализация с русскими символами"""
        return self.model_dump_json(indent=2)