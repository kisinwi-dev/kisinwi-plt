from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class SystemMessageType(Enum):
    INFO = "INFO"
    WARNING = "WARNING" 
    ERROR = "ERROR"

class AgentMessageType(Enum):
    START = "START"
    RESPONSE = "RESPONSE"
    
class SystemMessage(BaseModel):
    """Схема для создания сообщения от системы"""
    type_: SystemMessageType = Field(default_factory=SystemMessageType, description="Тип сообщения")
    message: str = Field(..., description="Сообщение от системы")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время события")
