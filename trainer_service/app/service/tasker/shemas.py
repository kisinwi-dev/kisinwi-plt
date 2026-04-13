from enum import Enum
from pydantic import BaseModel, Field

from app.core.task.models.shemas import ModelParams

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskParams(BaseModel):
    data_loader_params: dict = Field(..., description="Параметры данных")
    model_params: ModelParams = Field(..., description="Параметры используемой модели")
    trainer_params: dict = Field(..., description="Параметры обучения моделии")


class Task(BaseModel):
    """Схема задачи"""
    task_id: str = Field(..., description="ID задачи")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Статус задачи")
    params: TaskParams = Field(..., description="Параметры задачи")
