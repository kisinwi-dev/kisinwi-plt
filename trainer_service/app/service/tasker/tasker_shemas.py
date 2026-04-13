from pydantic import BaseModel, Field
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    """Схема задачи"""
    task_id: str = Field(..., description="ID задачи")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Статус задачи")
    payload: dict = Field(..., description="Параметры задачи")
