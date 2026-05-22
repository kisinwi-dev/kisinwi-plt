from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class ToolStatus(Enum):
    IN_PROGRESS = "IN PROGRESS"
    SUCCED = "SUCCEED"
    ERROR = "ERROR" 

class Tool(BaseModel):
    id: str = Field(..., description="Id инструмента")
    name: str = Field(..., description="Название инструмента")
    status: ToolStatus = Field(..., description="Статус")
    message: str = Field(..., description="Сообщение от системы")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время события")
