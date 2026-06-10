from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DataLoaderParams(BaseModel):
    dataset_id: str = Field(..., description="Id датасета")
    version_id: str = Field(..., description="Версия датасета")
    img_w_size: Optional[int] = Field(
        default=None,
        description="(deprecated, не используется — размер задаётся трансформациями) Ширина изображений"
    )
    img_h_size: Optional[int] = Field(
        default=None,
        description="(deprecated, не используется — размер задаётся трансформациями) Высота изображений"
    )
    batch_size: int = Field(default=32, description="Количество изображений в батче")
    num_workers: int = Field(
        default=2,
        ge=0,
        description="Количество процессов загрузки данных в DataLoader (0 — загрузка в основном процессе)"
    )

    # кастомная аугментация
    train_transforms_config: List[Dict[str, Any]] = Field(
        description="Конфигурация трансформаций для тренировочного датасета"
    )
    val_and_test_transforms_config: List[Dict[str, Any]] = Field(
        description="Конфигурация трансформаций для валидационного датасета"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_id": "dataset_123",
                "version_id": "v1.0",
                "batch_size": 32,
                "num_workers": 2,
                "train_transforms_config": [
                    {"name": "RandomResizedCrop", "params": {"size": [224, 224], "scale": [0.7, 1.0]}},
                    {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                    {"name": "RandomRotation", "params": {"degrees": 10}},
                    {"name": "ToTensor", "params": {}},
                    {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
                ],
                "val_and_test_transforms_config": [
                    {"name": "Resize", "params": {"size": [224, 224]}},
                    {"name": "ToTensor", "params": {}},
                    {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
                ]
            }
        }
    }