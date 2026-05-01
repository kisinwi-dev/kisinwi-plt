from pydantic import BaseModel, Field
from typing import Dict, Any, List

class Agent(BaseModel):
    name: str = Field(..., description="Имя агента")
    tools: List[str] = Field(..., description="Инструменты агента")

class AgentResponse(BaseModel):
    response_id: str = Field(..., description="ID ответа")
    agent: Agent = Field(..., description="Агент")
    out: str = Field(..., description="Ответ агента в виде текста")
    metrics: Dict[str, Any] = Field(..., description="Метрики использования агента")
