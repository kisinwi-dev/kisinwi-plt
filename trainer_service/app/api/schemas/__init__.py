from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ModelParams(BaseModel):
    type: str = Field("Resnet", description="Тип модели (Resnet)")
    name: str = Field("resnet18", description="Название модели (resnet18, resnet50 и т.д.)")
    weights: bool = Field(False, description="Использовать предобученные веса")


class LossFnConfig(BaseModel):
    type: str = Field("CrossEntropyLoss", description="Тип функции потерь")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"reduction": "mean"},
        description="Параметры функции потерь"
    )


class OptimizerConfig(BaseModel):
    type: str = Field("Adam", description="Тип оптимизатора")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "amsgrad": False
        },
        description="Параметры оптимизатора"
    )


class SchedulerConfig(BaseModel):
    type: str = Field(
        "OneCycleLR", 
        description="Тип планировщика (например, OneCycleLR)"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            'max_lr': 0.01,
            'total_steps': 100,
            'pct_start': 0.3,
        },
        description="Параметры планировщика"
    )


class DataLoaderParams(BaseModel):
    dataset_id: str = Field(..., description="Id датасета")
    version_id: str = Field(..., description="Id версии")
    img_w_size: int = Field(..., description="Ширина изображений после преобразований")
    img_h_size: int = Field(..., description="Высота изображений после преобразований")
    batch_size: int = Field(..., description="Размер батча")


class TrainerParams(BaseModel):
    loss_fn_config: LossFnConfig = Field(description="Конфигурация функции потерь")
    optimizer_config: OptimizerConfig = Field(..., description="Конфигурация оптимизатора")
    scheduler_config: SchedulerConfig = Field(...,description="Конфигурация планировщика")
    device: str = Field(..., description="Устройство для обучения (cpu/cuda)")
    epochs: int = Field(default=30, description="Количество эпох обучения")


class TaskConfig(BaseModel):
    task_id: str = Field(..., description="Id задачи")
    data_loader_params: DataLoaderParams
    model_params: ModelParams = Field(description="Параметры модели")
    trainer_params: TrainerParams = Field(description="Параметры тренера")

