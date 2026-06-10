from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

class MLModelStatus(BaseModel):
    """Информация о статусe модели"""
    id: int = Field(..., description="ID статуса")
    status: str = Field(..., description="Статус модели")
    description: Optional[str] = Field(None, description="Описание статуса")

class MLModelsCountByStatus(BaseModel):
    """Количество версий в конкретном статусе"""
    status: MLModelStatus = Field(..., description="Информация о статусе")
    count: int = Field(..., description="Количество версий в данном статусе")

class MLModelsStatistics(BaseModel):
    """Статистика по моделям"""
    total_models: int = Field(..., description="Общее количество моделей")
    total_versions: int = Field(..., description="Общее количество версий")
    by_status: List[MLModelsCountByStatus] = Field(..., description="Распределение версий по статусам")

class ModelVersion(BaseModel):
    """Полная схема версии модели (name/description денормализованы из родителя)"""
    id: str
    model_id: str = Field(..., description="ID родительской модели")
    name: str = Field(..., description="Имя модели")
    description: Optional[str] = Field(None, description="Описание модели")
    version: int = Field(..., description="Номер версии")
    model_type: str
    status: str = Field(..., description="Статус версии")
    metrics_report: str = Field(..., description="Описание метрик модели")
    classes: List[str] = Field(..., description="Список классов")
    train_params: Dict[str, Any] = Field(..., description="Параметры обучения модели")
    created_at: datetime = Field(..., description="Создание версии")

    # Датасет
    dataset_id: str = Field(..., description="Id датасета, на котором обучалась модель")
    dataset_version_id: str = Field(..., description="Id версии датасета, на котором обучалась модель")

    # Framework
    framework: Optional[str] = Field(None, description="Название framework для работы с моделью")
    framework_version: Optional[str] = Field(None, description="Версия framework")

class ModelVersions(BaseModel):
    """Плоский список версий с метаданными пагинации"""
    versions: List[ModelVersion] = Field(..., description="Список версий")
    total: int = Field(0, description="Общее количество версий с учётом фильтров")
    limit: Optional[int] = Field(None, description="Размер страницы из запроса (None - пагинация не применялась)")
    offset: int = Field(0, description="Смещение из запроса")

class Model(BaseModel):
    """Модель (родитель) со списком версий (убывающий порядок)"""
    id: str
    name: str = Field(..., description="Имя модели")
    description: Optional[str] = Field(None, description="Описание модели")
    created_at: datetime = Field(..., description="Создание модели")
    versions: List[ModelVersion] = Field(..., description="Версии модели (убывающий порядок)")

class Models(BaseModel):
    """Список моделей с метаданными пагинации (по моделям)"""
    models: List[Model] = Field(..., description="Список моделей")
    total: int = Field(0, description="Общее количество моделей с учётом фильтров")
    limit: Optional[int] = Field(None, description="Размер страницы из запроса")
    offset: int = Field(0, description="Смещение из запроса")

class ModelCreate(BaseModel):
    """Схема для создания модели"""
    name: str = Field(..., min_length=1, description="Имя модели (уникальное)")
    description: Optional[str] = None

class ModelUpdate(BaseModel):
    """Схема для обновления модели"""
    name: Optional[str] = Field(None, min_length=1, description="Новое имя модели")
    description: Optional[str] = None

class ModelVersionCreate(BaseModel):
    """Схема для создания версии модели (номер версии назначает сервер)"""
    model_type: str = Field(..., min_length=1, description="Тип модели")
    classes: List[str] = Field(..., min_length=1, description="Список классов (непустой)")
    train_params: Dict[str, Any]
    dataset_id: str
    dataset_version_id: str
    framework: Optional[str] = None
    framework_version: Optional[str] = None

class ModelVersionUpdate(BaseModel):
    """Схема для обновления версии модели"""
    status: Optional[str] = None
    metrics_report: Optional[str] = None
    train_params: Optional[Dict[str, Any]] = None
    framework: Optional[str] = None
    framework_version: Optional[str] = None
