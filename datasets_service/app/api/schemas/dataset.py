from datetime import datetime
from typing import Dict, List, Literal, Optional
from enum import Enum
from pydantic import (
    BaseModel, Field,
    HttpUrl, model_validator
)

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

class Source(BaseModel):
    type: Literal['kaggle', 'url', 'huggingface', 'other']
    url: Optional[HttpUrl] = Field(None, description='URL страницы с информацией о датасете')
    description: str

    @model_validator(mode="after")
    def validate_url_for_type(self):
        """Проверяет, что url заполнен для типов, где он обязателен"""
        if self.type in ['kaggle', 'huggingface', 'url'] and not self.url:
            raise ValueError(f"Для источника типа '{self.type}' необходимо указать url")
        return self

class Version(BaseModel):
    version_id: str
    description: str = Field(..., description="Описание версии")
    num_samples: int = Field(..., ge=0, description="Количество изображений")
    size_bytes: int = Field(..., ge=0, description="Вес версии в байтах")

    image_format_stats: Dict[str, int] = Field(default_factory=dict, description="Формат изображений")
    splits: Dict[SplitType, Split] = Field(default_factory=dict, description='Выборки test/trai/val')

    created_at: datetime = Field(default_factory=datetime.now, frozen=True, description="Время создания")

    @model_validator(mode="after")
    def check_split_consistency(self):
        count_img = 0
        for split_name, split in self.splits.items():
            split_total = sum(class_.count for class_ in split.class_distribution)

            # Проверяем, что split_name корректен
            if split_name not in SplitType:
                raise ValueError(f"Неизвестный тип сплита: {split_name}")

            count_img += split_total

        if count_img != self.num_samples:
            raise ValueError(
                f"Сумма изображений по всем сплитам ({count_img}) != num_samples ({self.num_samples})"
            )
        return self

class DatasetMetadata(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    name: str
    description: str
    num_classes: int = Field(..., ge=1)
    class_names: List[str] = Field(..., min_length=1)
    class_to_idx: Dict[str, int]
    sources: List[Source]
    type: Literal["image", "text", "tabular", "other"] = "image"
    task: Literal["classification", "regression", "detection", "segmentation", "other"] = "classification"
    default_version_id: str
    versions: List[Version] = Field(..., min_length=1)

    created_at: datetime = Field(default_factory=datetime.now, frozen=True)
    updated_at: datetime = Field(default_factory=datetime.now, frozen=True)

    model_config = {
        "validate_assignment": True
    }

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        object.__setattr__(self, "updated_at", datetime.now())

    @model_validator(mode="after")
    def validate_class_mapping(self):

        if set(self.class_names) != set(self.class_to_idx.keys()):
            raise ValueError(
                "class_to_idx ключи должны совпадать с именами классов"
            )

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "num_classes должно быть равно количеству class_names"
            )

        return self
    
    @model_validator(mode="after")
    def validate_default_version(self):

        version_ids = [v.version_id for v in self.versions]

        if self.default_version_id not in version_ids:
            raise ValueError(
                "default_version_id должен существовать в версиях"
            )

        return self