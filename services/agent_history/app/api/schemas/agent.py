import uuid
from datetime import datetime
from pydantic import BaseModel, Field, UUID4

class AgentResponseCreate(BaseModel):
    """Модель ответа агента (Pydantic схема)"""
    discussion_id: str = Field(..., description="ID дискуссии")
    agent_response: AgentResponse = Field(..., description="Информация об ответе агента")

class AgentResponse(BaseModel):
    """Модель ответа агента (Pydantic схема)"""
    
    agent_role: str = Field(..., description="Роль агента")
    text: str = Field(..., description="Текст ответа агента", min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")
    response_id: UUID4 = Field(default_factory=uuid.uuid4, description="Уникальный ID ответа")
    
    def model_dump_json_custom(self) -> str:
        """Кастомная сериализация с русскими символами"""
        return self.model_dump_json(indent=2)