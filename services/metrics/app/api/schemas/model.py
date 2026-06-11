from pydantic import BaseModel, Field
from typing import List, Any, Literal, Optional

Split = Literal["train", "val", "test"]

class ModelMetricData(BaseModel):
    """Схема метрик"""
    name: str = Field(..., description="Название метрики")
    split: Optional[Split] = Field(
        None,
        description="Выборка (train/val/test); если не задана — выводится из префикса названия"
    )
    values: List[Any] = Field(default_factory=list, description="Значения метрик")

class ModelMetricAdd(BaseModel):
    """Схема на добавление метрики"""
    model_id: str = Field(..., description="ID модели")
    metric: ModelMetricData = Field(..., description="Метрика")

class ModelMetricAdds(BaseModel):
    """Схема на добавление нескольких метрик"""
    model_id: str = Field(..., description="ID модели")
    metrics: List[ModelMetricData] = Field(default_factory=list, description="Список метрик")

class ModelMetrics(BaseModel):
    """ID модели и её метрики, разбитые по выборкам"""
    model_id: str = Field(..., description="ID модели")
    train: List[ModelMetricData] = Field(default_factory=list, description="Метрики тренировочной выборки")
    val: List[ModelMetricData] = Field(default_factory=list, description="Метрики валидационной выборки")
    test: List[ModelMetricData] = Field(default_factory=list, description="Метрики тестовой выборки")

class ModelMetricsBatchRequest(BaseModel):
    """Запрос метрик сразу нескольких моделей"""
    model_ids: List[str] = Field(default_factory=list, description="Список ID моделей")
