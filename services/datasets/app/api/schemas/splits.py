from enum import Enum
from typing import Dict, List
from pydantic import BaseModel, Field

class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class ClassDistribution(BaseModel):
    """Распределение классов для одного сплита"""
    class_name: str = Field(..., description="Имя класса")
    class_id: int = Field(..., description="Id класса")
    count: int = Field(..., description="Количество обьектов")
    percentage: float = Field(..., description="Процент от общего количества в выборке")
    image_size_count: Dict[str, int] = Field(..., description="Количество изображений, с определённым размером")

class Split(BaseModel):
    class_distribution: List[ClassDistribution] = Field(default_factory=list, description="Информации про каждому классу")
