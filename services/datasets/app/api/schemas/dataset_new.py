from typing import Literal, List
from pydantic import BaseModel, Field

from .dataset import Source

class NewVersion(BaseModel):
    id_data: str = Field(description="ID загруженных данных")
    name: str = Field(description="Название версии")
    description: str = Field("Нет описания", description="Описание. Особенности данных.")
    sources: List[Source] = Field(description="Информация о ресурсах откуда брали данные.")

class NewDataset(BaseModel):
    name: str = Field(description="Название датасета")
    description: str = Field(description="Описание датасета. Зачем он нужен и какие задачи должен решить датасет.")
    type: Literal["image", "text", "tabular", "other"] = Field("image", description="Тип данных изображения/текст/таблицы")
    task: Literal["classification", "regression", "detection", "segmentation", "other"] = Field(
        "classification", 
        description="Задача решаемая датасетом классификация/регрессия/детекцмя"
    )
    version: NewVersion = Field(description="Новая версия")