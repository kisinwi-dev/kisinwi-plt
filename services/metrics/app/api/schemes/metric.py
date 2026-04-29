from pydantic import BaseModel, Field
from typing import List

class TrainingMetricAdd(BaseModel):
    """Схема на добавление метрики"""
    task_id: str = Field(..., description='ID задачи')
    name: str = Field(..., description='Название метрики')
    value: float = Field(..., description='Значения метрики')

class TrainingMetricAdds(BaseModel):
    """Схема на добавление нескольких метрик"""
    task_id: str = Field(..., description='ID задачи')
    metrics: List[TrainingMetricData] = Field(..., description='Список метрик')

class TrainingMetricData(BaseModel):
    """Схема метрик"""
    name: str = Field(..., description='Название метрики')
    values: List[float] = Field(default_factory=list, description='Значения метрик')

class TaskTrainingMetrics(BaseModel):
    """Задача и все её метрики"""
    task_id: str = Field(...,description='Id задачи')
    metrics: List[TrainingMetricData] = Field(default_factory=list, description='Метрики')

class SearchTrainingMetric(BaseModel):
    """Запрос на поиск требуемой метрики"""
    task_id: str = Field(..., description='Id задачи')
    name: str = Field(..., description='Название метрики')