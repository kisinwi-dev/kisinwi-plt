from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class AgentResponse(BaseModel):
    response_id: str = Field(..., description="ID ответа")
    discussion_id: Optional[str] = Field(None, description="ID дискуссии, к которой относится ответ")
    metrics: Dict[str, Any] = Field(..., description="Метрики использования агента")

class AgentDiscussionMetrics(BaseModel):
    """Метрики всех агентов дискуссии и их сводка"""
    discussion_id: str = Field(..., description="ID дискуссии")
    responses: List[AgentResponse] = Field(default_factory=list, description="Метрики каждого ответа агента")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Суммарные метрики по дискуссии")
