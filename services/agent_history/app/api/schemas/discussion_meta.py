from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DiscussionStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class DiscussionMeta(BaseModel):
    discussion_id: str = Field(..., description="ID дискуссии")
    title: Optional[str] = Field(None, description="Название дискуссии")
    status: DiscussionStatus = Field(DiscussionStatus.ACTIVE, description="Статус дискуссии")
    tags: list[str] = Field(default_factory=list, description="Теги")
    pipeline: Optional[str] = Field(None, description="Название pipeline")
    agent_roles: list[str] = Field(default_factory=list, description="Роли агентов в дискуссии")
    created_at: datetime = Field(default_factory=datetime.now, description="Время старта дискуссии")
    finished_at: Optional[datetime] = Field(None, description="Время завершения дискуссии")


class AgentModelInfo(BaseModel):
    """Агент дискуссии и использованные им модели LLM."""
    role: str = Field(..., description="Роль агента")
    models: list[str] = Field(default_factory=list, description="Модели LLM, которыми отвечал агент")


class DiscussionMetaRead(DiscussionMeta):
    """Метаданные дискуссии с вычисляемыми агрегатами для списка."""
    responses_count: int = Field(0, description="Количество ответов агентов (длина диалога)")
    tool_calls_count: int = Field(0, description="Количество вызовов инструментов")
    agents: list[AgentModelInfo] = Field(default_factory=list, description="Агенты дискуссии и их модели LLM")


class DiscussionMetaUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[DiscussionStatus] = None
    tags: Optional[list[str]] = None
    pipeline: Optional[str] = None
    agent_roles: Optional[list[str]] = None


class CreateDiscussion(BaseModel):
    discussion_id: Optional[str] = Field(None, description="ID дискуссии. Генерируется если не передан")
    title: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    pipeline: Optional[str] = None
    agent_roles: list[str] = Field(default_factory=list)
