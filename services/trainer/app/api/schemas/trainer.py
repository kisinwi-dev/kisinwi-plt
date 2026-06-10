from typing import Dict, List, Literal
from pydantic import BaseModel, Field

class LossConfig(BaseModel):
    name: str = Field(
        ..., 
        description="Название функции потерь из torch.nn (например, 'CrossEntropyLoss', 'MSELoss')"
    )
    params: Dict = Field(
        ..., 
        description="Параметры функции потерь (например, {'reduction': 'mean', 'label_smoothing': 0.1})"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "CrossEntropyLoss",
                "params": {
                    "reduction": "mean",
                    "label_smoothing": 0.1
                }
            }
        }
    }

class OptimizerConfig(BaseModel):
    name: str = Field(
        ..., 
        description="Название оптимизатора из torch.optim (например, 'Adam', 'AdamW', 'SGD')"
    )
    params: Dict = Field(
        ..., 
        description="Параметры оптимизатора (например, {'lr': 0.001, 'weight_decay': 0.0001})"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "AdamW",
                "params": {
                    "lr": 0.0001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.05,
                    "amsgrad": False
                }
            }
        }
    }

class SchedulerConfig(BaseModel):
    name: str = Field(
        ..., 
        description="Название планировщика из torch.optim.lr_scheduler (например, 'CosineAnnealingLR', 'StepLR')"
    )
    params: Dict = Field(..., description="Параметры планировщика (например, {'T_max': 30, 'eta_min': 1e-6})")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "CosineAnnealingLR",
                "params": {
                    "T_max": 30,
                    "eta_min": 1e-6,
                    "last_epoch": -1
                }
            }
        }
    }

class EarlyStop(BaseModel):
    metric_name: str = Field(
        default='loss',
        description='Метрикa на которую стоит смотреть при обучении'
    )
    patience: int = Field(
        default=4,
        description="Количество эпох по которым мы опрделяем изменяемость"
    )
    min_delta: float = Field(
        default=0.001,
        description="Размер допустимого изменения метрики"
    )
    mode: Literal['min', 'max'] = Field(
        default='min',
        description="Направление улучшения метрики: 'min' (например, loss) или 'max' (например, accuracy)"
    )

class TrainerParams(BaseModel):
    loss_fn: LossConfig = Field(
        ..., 
        description="Конфигурация функции потерь для оптимизации модели"
    )
    optimizer: OptimizerConfig = Field(
        ..., 
        description="Конфигурация оптимизатора для обновления весов"
    )
    scheduler: SchedulerConfig = Field(
        ...,
        description="Конфигурация планировщика скорости обучения"
    )
    epochs: int = Field(
        ...,
        description="Количество эпох обучения"
    )
    early_stop: EarlyStop = Field(
        default_factory=EarlyStop,
        description="Параметры ранней остановки обучения"
    )
    use_amp: bool = Field(
        default=False,
        description="Automatic Mixed Precision: ускоряет обучение на GPU, на CPU игнорируется"
    )
    grad_clip_norm: float | None = Field(
        default=None,
        description="Максимальная норма градиентов (clip_grad_norm_); None — без ограничения"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "loss_fn": {
                    "name": "CrossEntropyLoss",
                    "params": {
                        "reduction": "mean",
                        "label_smoothing": 0.1
                    }
                },
                "optimizer": {
                    "name": "AdamW",
                    "params": {
                        "lr": 0.001,
                        "weight_decay": 0.01,
                        "betas": [0.9, 0.999]
                    }
                },
                "scheduler": {
                    "name": "CosineAnnealingLR",
                    "params": {
                        "T_max": 50,
                        "eta_min": 1e-6
                    }
                },
                "epochs": 50,
                "early_stop": {
                    "metric_name": "loss",
                    "patience": 4,
                    "min_delta": 0.001,
                    "mode": "min"
                },
                "use_amp": False,
                "grad_clip_norm": 1.0
            }
        }
    }
