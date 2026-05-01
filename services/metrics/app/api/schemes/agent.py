from pydantic import BaseModel, Field
from typing import Dict, Any

class AgentResponse(BaseModel):
    response_id: str = Field(..., description="ID ответа")
    metrics: Dict[str, Any] = Field(..., description="Метрики использования агента")
