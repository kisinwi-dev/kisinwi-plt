from typing import List, Optional
from pydantic import BaseModel, Field

class MetricesParams(BaseModel):
    metrics_list: List[str] = Field(
        default=['accuracy', 'precision', 'recall', 'f1'],
        description='Список метрик (loss добавляется автоматически)'
    )
    average: Optional[str] = Field(
        default='macro',
        description='Тип усреднения precision/recall/f1'
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "metrics_list": ['accuracy', 'precision', 'recall', 'f1'],
                "average": 'macro'
            }
        }
    }

class MetricesParamCollections(BaseModel):
    train: MetricesParams = Field(
        ...,
        description="Метрики для обучения"
    )
    val: MetricesParams = Field(
        ...,
        description="Метрики для валидации"
    )
    test: MetricesParams = Field(
        ...,
        description="Метрики для тестирования"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "train": {
                    "metrics_list": ['accuracy', 'loss'],
                    "average": 'macro'
                },
                "val": {
                    "metrics_list": ['accuracy', 'precision', 'recall', 'f1'],
                    "average": 'macro'
                },
                "test": {
                    "metrics_list": ['accuracy', 'precision', 'recall', 'f1', 'specificity'],
                    "average": 'macro'
                }
            }
        }
    }