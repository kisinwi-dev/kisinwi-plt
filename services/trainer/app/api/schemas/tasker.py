from enum import Enum
from pydantic import BaseModel, Field

from .ml_models import ModelParams
from .data import DataLoaderParams
from .trainer import TrainerParams
from .metrices import MetricesParamCollections

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

    model_config = {
        "json_schema_extra": {
            "example": {
                "data_loader_params": {
                    "dataset_id": "dataset_123",
                    "version_id": "v1.0",
                    "img_w_size": 224,
                    "img_h_size": 224,
                    "batch_size": 32,
                    "is_calculate_normalize_dataset": False
                },
                "model_params": {
                    "type": "resnet50",
                    "pretrained": True
                },
                "metrices_params": {
                    "train": {
                        "metrics_list": ["accuracy", "loss"],
                        "average": "macro"
                    },
                    "val": {
                        "metrics_list": ["accuracy", "precision", "recall", "f1"],
                        "average": "macro"
                    },
                    "test": {
                        "metrics_list": ["accuracy", "precision", "recall", "f1"],
                        "average": "macro"
                    }
                },
                "trainer_params": {
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
                            "weight_decay": 0.01
                        }
                    },
                    "scheduler": {
                        "name": "CosineAnnealingLR",
                        "params": {
                            "T_max": 50,
                            "eta_min": 1e-6
                        }
                    },
                    "epochs": 50
                },
                "device": "cuda:0"
            }
        }
    }


class Task(BaseModel):
    """Схема задачи"""
    task_id: str = Field(..., description="ID задачи")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Статус задачи")
    params: TaskParams = Field(..., description="Параметры задачи")
