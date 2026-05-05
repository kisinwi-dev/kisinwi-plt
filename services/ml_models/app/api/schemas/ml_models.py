from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from typing import Optional, List, Dict, Any

class MLModelsStatistics(BaseModel):
    """Статситика по моделям"""
    total: int = Field(..., description="Общее поличество моделей")
    by_status: List[MLModelsCountByStatus] = Field(..., description="Распределение моделей по статусам")

class MLModelsCountByStatus(BaseModel):
    """Информация о количестве моделей с """
    status: MLModelStatus = Field(..., description="Информация о статусе")
    count: int = Field(..., description="Количество моделей в данном статусе")

class MLModelStatus(BaseModel):
    """Информация о статусe модели"""
    id: int = Field(..., description="ID статуса")
    status: str = Field(..., description="Статус модели")
    description: Optional[str] = Field(None, description="Описание статуса")

class MLModel(BaseModel):
    """Полная схема ML модели"""
    id: UUID
    name: str = Field(..., description="Имя модели")
    version: int = Field(..., description="Версия модели")
    model_type: str
    description: Optional[str] = Field(None, description="Описание модели")
    classes: List[str] = Field(..., description="Список классов")
    train_params: Dict[str, Any] = Field(
        default_factory=Dict[str, Any], 
        description="Параметры обучения модели"
    )
    created_at: datetime = Field(..., description="Создание модели")

    # Датасет
    dataset_id: UUID = Field(..., description="Id датасета, на котором обучалась модель")
    dataset_version_id: str = Field(..., description="Id версии датасета, на котором обучалась модель")
    
    # Framework
    framework: Optional[str] = Field(None, description="Название framework для работы с моделью")
    framework_version: Optional[str] = Field(None, description="Версия framework")
    
    # Путь до весов
    storage_path: Optional[str] = Field(None, description="Путь до весов модели")

class MLModelCreate(BaseModel):
    """Схема для создания ML модели"""
    name: str
    version: int
    model_type: str
    description: Optional[str] = None
    classes: List[str]
    train_params: Dict[str, Any]
    dataset_id: UUID
    dataset_version_id: str
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    storage_path: Optional[str] = None

class MLModelUpdate(BaseModel):
    """Схема для обновления ML модели"""
    name: Optional[str] = None
    version: Optional[int] = None
    model_type: Optional[str] = None
    description: Optional[str] = None
    classes: Optional[List[str]] = None
    train_params: Optional[Dict[str, Any]] = None
    dataset_id: Optional[UUID] = None
    dataset_version_id: Optional[str] = None
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    storage_path: Optional[str] = None