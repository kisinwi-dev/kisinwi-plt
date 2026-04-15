from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field

from app.core.models import ModelParams
from app.core.datas import DataLoaderParams
from app.service.metrices.shemas import MetricesParamCollections

class LossConfig(BaseModel):
    name: str = Field(
        ..., 
        description="Название функции потерь из torch.nn (например, 'CrossEntropyLoss', 'MSELoss')"
    )
    params: Dict = Field(
        ..., 
        description="Параметры функции потерь (например, {'reduction': 'mean', 'label_smoothing': 0.1})"
    )

class OptimizerConfig(BaseModel):
    name: str = Field(
        ..., 
        description="Название оптимизатора из torch.optim (например, 'Adam', 'AdamW', 'SGD')"
    )
    params: Dict = Field(
        ..., 
        description="Параметры оптимизатора (например, {'lr': 0.001, 'weight_decay': 0.0001})"
    )

class ShedulerConfig(BaseModel):
    name: str = Field(
        ..., 
        description="Название планировщика из torch.optim.lr_scheduler (например, 'CosineAnnealingLR', 'StepLR')"
    )
    params: Dict = Field(..., description="Параметры планировщика (например, {'T_max': 30, 'eta_min': 1e-6})")

class TrainerParams(BaseModel):
    loss_fn: LossConfig = Field(
        ..., 
        description="Конфигурация функции потерь для оптимизации модели"
    )
    optimizer: OptimizerConfig = Field(
        ..., 
        description="Конфигурация оптимизатора для обновления весов"
    )
    scheduler: ShedulerConfig = Field(
        ..., 
        description="Конфигурация планировщика скорости обучения"
    )
    epochs: int = Field(
        ..., 
        description="Количество эпох обучения"
    )

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskParams(BaseModel):
    data_loader_params: DataLoaderParams = Field(..., description="Параметры данных")
    model_params: ModelParams = Field(..., description="Параметры используемой модели")
    metrices_params: MetricesParamCollections = Field(..., description="Параметры используемой коллекции метрик")
    trainer_params: TrainerParams = Field(..., description="Параметры обучения моделии")
    device: str = Field(..., description="На каком устройстве проходит вычисление")


class Task(BaseModel):
    """Схема задачи"""
    task_id: str = Field(..., description="ID задачи")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Статус задачи")
    params: TaskParams = Field(..., description="Параметры задачи")
