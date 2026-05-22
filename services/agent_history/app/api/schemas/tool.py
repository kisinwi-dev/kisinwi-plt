from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class ToolType(Enum):
    CALL = "CALL"
    RESULT = "RESULT"

class ToolStatus(Enum):
    IN_PROGRESS = "IN PROGRESS"
    SUCCED = "SUCCEED"
    ERROR = "ERROR" 

class Tool(BaseModel):
    id: str = Field(..., description="Id инструмента")
    type_: ToolType = Field(..., description="Тип")
    status: ToolStatus = Field(..., description="Статус")
    message: str = Field(..., description="Сообщение от системы")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время события")
