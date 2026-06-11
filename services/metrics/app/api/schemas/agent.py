from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class AgentResponse(BaseModel):
    response_id: str = Field(..., description="ID ответа", examples=["resp-7f3a"])
    discussion_id: Optional[str] = Field(
        None,
        description="ID дискуссии, к которой относится ответ",
        examples=["disc-12"],
    )
    metrics: Dict[str, Any] = Field(
        ...,
        description="Метрики использования агента",
        examples=[{"prompt_tokens": 784, "completion_tokens": 450, "time_ms": 5600}],
    )

class AgentDiscussionMetrics(BaseModel):
    """Метрики всех агентов дискуссии и их сводка"""
    discussion_id: str = Field(..., description="ID дискуссии", examples=["disc-12"])
    responses: List[AgentResponse] = Field(default_factory=list, description="Метрики каждого ответа агента")
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Суммарные метрики по дискуссии",
        examples=[{"prompt_tokens": 1568, "completion_tokens": 902, "time_ms": 11200}],
    )

class AgentAddResponse(BaseModel):
    """Результат добавления метрик ответа агента"""
    response_id: str = Field(..., description="ID ответа", examples=["resp-7f3a"])
    added: bool = Field(..., description="Признак добавления метрик")

class AgentDeleteResponse(BaseModel):
    """Результат удаления метрик ответа агента"""
    response_id: str = Field(..., description="ID ответа", examples=["resp-7f3a"])
    deleted: bool = Field(..., description="Признак удаления метрик")
