from pydantic import BaseModel, Field
from typing import List, Any

class ModelMetricData(BaseModel):
    """Схема метрик"""
    name: str = Field(..., description="Название метрики")
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
    """ID модели и все её метрики"""
    model_id: str = Field(..., description="ID модели")
    metrics: List[ModelMetricData] = Field(default_factory=list, description="Список метрик")

class SearchMetric(BaseModel):
    """Запрос на поиск требуемой метрики"""
    model_id: str = Field(..., description="ID модели")
    name: str = Field(..., description="Название метрики")