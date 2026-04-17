from pydantic import BaseModel, Field
from typing import List

class MetricAdd(BaseModel):
    """Схема на добавление метрики"""
    task_id: str = Field(..., description='ID задачи')
    metric_name: str = Field(..., description='Название метрики')
    value: float = Field(..., description='Значения метрики')

class MetricData(BaseModel):
    """Схема метрик"""
    metric_name: str = Field(..., description='Название метрики')
    values: List[float] = Field(default_factory=list, description='Значения метрик')

class TaskMetrics(BaseModel):
    """Задача и все её метрики"""
    task_id: str = Field(...,description='Id задачи')
    metrics: List[MetricData] = Field(default_factory=list, description='Метрики')

class SearchMetric(BaseModel):
    """Запрос на поиск требуемой метрики"""
    task_id: str = Field(..., description='Id задачи')
    metric_name: str = Field(..., description='Название метрики')