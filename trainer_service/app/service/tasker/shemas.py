from enum import Enum
from pydantic import BaseModel, Field

from app.core.models import ModelParams
from app.core.datas import DataLoaderParams
from app.service.metrices.shemas import MetricesParamCollections

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskParams(BaseModel):
    data_loader_params: DataLoaderParams = Field(..., description="Параметры данных")
    model_params: ModelParams = Field(..., description="Параметры используемой модели")
    metrices_params: MetricesParamCollections = Field(..., description="Параметры используемой коллекции метрик")
    trainer_params: dict = Field(..., description="Параметры обучения моделии")
    device: str = Field(..., description="На каком устройстве проходит вычисление")


class Task(BaseModel):
    """Схема задачи"""
    task_id: str = Field(..., description="ID задачи")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Статус задачи")
    params: TaskParams = Field(..., description="Параметры задачи")
