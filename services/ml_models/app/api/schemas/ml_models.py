from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

class MLModelStatus(BaseModel):
    """Информация о статусe модели"""
    id: int = Field(..., description="ID статуса")
    status: str = Field(..., description="Статус модели")
    description: Optional[str] = Field(None, description="Описание статуса")

class MLModelsCountByStatus(BaseModel):
    """Количество моделей в конкретном статусе"""
    status: MLModelStatus = Field(..., description="Информация о статусе")
    count: int = Field(..., description="Количество моделей в данном статусе")

class MLModelsStatistics(BaseModel):
    """Статистика по моделям"""
    total: int = Field(..., description="Общее количество моделей")
    by_status: List[MLModelsCountByStatus] = Field(..., description="Распределение моделей по статусам")

class MLModels(BaseModel):
    """ML модели с метаданными пагинации"""
    models: List[MLModel] = Field(..., description="Список ML моделей")
    total: int = Field(0, description="Общее количество моделей с учётом фильтров")
    limit: Optional[int] = Field(None, description="Размер страницы из запроса (None - пагинация не применялась)")
    offset: int = Field(0, description="Смещение из запроса")

class MLModel(BaseModel):
    """Полная схема ML модели"""
    id: str
    name: str = Field(..., description="Имя модели")
    version: int = Field(..., description="Версия модели")
    model_type: str
    status: str = Field(..., description="Статус модели")
    description: Optional[str] = Field(None, description="Описание модели")
    metrics_report: str = Field(..., description="Описание метрик модели")
    classes: List[str] = Field(..., description="Список классов")
    train_params: Dict[str, Any] = Field(..., description="Параметры обучения модели")
    created_at: datetime = Field(..., description="Создание модели")

    # Датасет
    dataset_id: str = Field(..., description="Id датасета, на котором обучалась модель")
    dataset_version_id: str = Field(..., description="Id версии датасета, на котором обучалась модель")
    
    # Framework
    framework: Optional[str] = Field(None, description="Название framework для работы с моделью")
    framework_version: Optional[str] = Field(None, description="Версия framework")

class MLModelCreate(BaseModel):
    """Схема для создания ML модели"""
    name: str = Field(..., min_length=1, description="Имя модели")
    version: int = Field(..., ge=1, description="Версия модели (>= 1)")
    model_type: str = Field(..., min_length=1, description="Тип модели")
    description: Optional[str] = None
    classes: List[str] = Field(..., min_length=1, description="Список классов (непустой)")
    train_params: Dict[str, Any]
    dataset_id: str
    dataset_version_id: str
    framework: Optional[str] = None
    framework_version: Optional[str] = None

class MLModelUpdate(BaseModel):
    """Схема для обновления ML модели"""
    status: Optional[str] = None
    description: Optional[str] = None
    metrics_report: Optional[str] = None
    train_params: Optional[Dict[str, Any]] = None
    framework: Optional[str] = None
    framework_version: Optional[str] = None

class MLModelGroup(BaseModel):
    """Группа версий одной модели"""
    name: str = Field(..., description="Имя модели")
    versions: List[MLModel] = Field(..., description="Версии модели (убывающий порядок)")

class MLModelsGrouped(BaseModel):
    """Сгруппированный список моделей с пагинацией по уникальным именам"""
    groups: List[MLModelGroup] = Field(..., description="Группы моделей")
    total: int = Field(0, description="Общее количество уникальных имён с учётом фильтров")
    limit: Optional[int] = Field(None, description="Размер страницы (по именам)")
    offset: int = Field(0, description="Смещение из запроса")
