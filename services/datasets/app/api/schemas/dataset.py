from datetime import datetime
from typing import Dict, List, Literal, Optional
from pydantic import (
    BaseModel, Field,
    HttpUrl, model_validator
)
from .splits import (
    Split, SplitType, SplitSummaryResponse,
    SplitCountsResponse, SplitBalanceInfo, SplitBalanceResponse,
    ClassDistributionItem, ClassDistributionResponse,
    ImageSizeStats, ImageSizeStatsResponse
)

class Source(BaseModel):
    type: Literal["kaggle", "url", "huggingface", "other"]
    url: Optional[HttpUrl] = Field(None, description="URL страницы с информацией о датасете")
    description: str = Field("Нет описания", description="Описание ресурса")

    @model_validator(mode="after")
    def validate_url_for_type(self):
        """Проверяет, что url заполнен для типов, где он обязателен"""
        if self.type in ["kaggle", "huggingface", "url"] and not self.url:
            raise ValueError(f"Для источника типа '{self.type}' необходимо указать url")
        return self

class VersionResponse(BaseModel):
    id: str
    name: str = Field(description="Название версии")
    description: str = Field(..., description="Описание версии")
    sources: List[Source] = Field(description="Ресурсы данных")
    num_samples: int = Field(..., ge=0, description="Количество изображений")
    size_bytes: int = Field(..., ge=0, description="Вес версии в байтах")
    image_format_stats: Dict[str, int] = Field(default_factory=dict, description="Формат изображений")
    created_at: datetime = Field(default_factory=datetime.now, frozen=True, description="Время создания")

class Version(BaseModel):
    id: str
    name: str = Field(description="Название версии")
    description: str = Field(..., description="Описание версии")
    sources: List[Source] = Field(description="Ресурсы данных")
    num_samples: Optional[int] = Field(default=None, ge=0, description="Количество изображений")
    size_bytes: int = Field(..., ge=0, description="Вес версии в байтах")

    image_format_stats: Dict[str, int] = Field(default_factory=dict, description="Формат изображений")
    splits: Dict[SplitType, Split] = Field(default_factory=dict, description='Выборки test/trai/val')

    created_at: datetime = Field(default_factory=datetime.now, frozen=True, description="Время создания")

    @model_validator(mode="after")
    def check_split_consistency(self):
        if self.num_samples is None:
            self.num_samples = 0
            for split_name, split in self.splits.items():
                self.num_samples += sum(class_.count for class_ in split.class_distribution)

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
    
    def get_version_response(self) -> VersionResponse:
        return VersionResponse(**self.model_dump(exclude={'splits'}))
    
    def get_split_summary(self) -> SplitSummaryResponse:
        """Возвращает сводку по сплитам"""

        # статистика размеров изображений 
        image_size_stats = {}
        # инфо по сплиту 
        splits_summary = {}

        for split_type, split in self.splits.items():
            image_size_stats[split_type.value] = split.to_image_size_summary()
            splits_summary[split_type.value] = split.to_summary()

        return SplitSummaryResponse(
            id=self.id,
            name=self.name,
            num_samples=self.num_samples,
            size_bytes=self.size_bytes,
            splits_summary=splits_summary,
            image_size_stats=image_size_stats,
            overall_balance=self._get_overall_balance()
        )
        
    def get_split_counts(self) -> SplitCountsResponse:
        """Возвращает количество изображений по сплитам"""
        return SplitCountsResponse(
            id=self.id,
            name=self.name,
            num_samples=self.num_samples,
            counts_per_split={
                split_type.value: split.total_samples
                for split_type, split in self.splits.items()
            }
        )

    def get_split_balance(self) -> SplitBalanceResponse:
        """Возвращает баланс классов по сплитам"""
        return SplitBalanceResponse(
            id=self.id,
            name=self.name,
            overall_balance=self._get_overall_balance(),
            splits={
                split_type.value: SplitBalanceInfo(
                    total_samples=split.total_samples,
                    num_classes=split.num_classes,
                    balance_ratio=split.get_balance_ratio(),
                    is_balanced=split.is_balanced()
                )
                for split_type, split in self.splits.items()
            }
        )

    def get_class_distribution_response(self) -> ClassDistributionResponse:
        """Возвращает распределение классов по сплитам"""
        return ClassDistributionResponse(
            id=self.id,
            name=self.name,
            splits={
                split_type.value: [
                    ClassDistributionItem(
                        class_name=cd.class_name,
                        class_id=cd.class_id,
                        count=cd.count,
                        percentage=cd.percentage
                    )
                    for cd in split.class_distribution
                ]
                for split_type, split in self.splits.items()
            }
        )

    def get_image_size_stats(self) -> ImageSizeStatsResponse:
        """Возвращает статистику размеров изображений по сплитам"""
        return ImageSizeStatsResponse(
            id=self.id,
            name=self.name,
            splits={
                split_type.value: ImageSizeStats(**split.to_image_size_summary())
                for split_type, split in self.splits.items()
            }
        )

    def _get_overall_balance(self) -> float:
        """Общий баланс по всем сплитам"""
        all_counts = {}
        for split in self.splits.values():
            for cd in split.class_distribution:
                all_counts[cd.class_name] = all_counts.get(cd.class_name, 0) + cd.count
        
        if not all_counts:
            return 0.0
        
        counts = list(all_counts.values())
        return min(counts) / max(counts)
    
    def get_split(self, split_type: SplitType) -> Optional[Split]:
        """Получить конкретный сплит"""
        return self.splits.get(split_type)
    
    def get_class_distribution(self, split_type: SplitType) -> List[Dict]:
        """Получить распределение классов для конкретного сплита"""
        split = self.get_split(split_type)
        if not split:
            return []
        return [cd.to_summary() for cd in split.class_distribution]
    
class DatasetResponse(BaseModel):
    id: str = Field(..., min_length=1, description="Id датасета")
    name: str = Field(description="Название датасета")
    description: str = Field(description="Описание датасета")
    
    classes_count: int = Field(..., ge=1)
    classes_names: List[str] = Field(..., min_length=1)
    classes_to_idx: Dict[str, int] = Field(description="Словарь, где ключ название класса, а значение его индекс")

    type: Literal["image", "text", "tabular", "other"] = Field(description="Тип данных")
    task: Literal["classification", "regression", "detection", "segmentation", "other"] = Field(description="Название задачи")
    default_version_id: str = Field(description="ID версим являющейся стандартным значением")
    versions: List[VersionResponse] = Field(min_length=1, description="Список версий")

    created_at: datetime = Field(default_factory=datetime.now, frozen=True, description="Время создания датасета")
    updated_at: datetime = Field(default_factory=datetime.now, frozen=True, description="Время Изменения датасета")

    model_config = {
        "validate_assignment": True
    }

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        object.__setattr__(self, "updated_at", datetime.now())
    
    @model_validator(mode="after")
    def validate_default_version(self):

        version_ids = [v.id for v in self.versions]

        if self.default_version_id not in version_ids:
            raise ValueError(
                "Стандартная версия должна существовать в списке версий"
            )

        return self

class DatasetMetadata(BaseModel):
    id: str = Field(..., min_length=1, description="Id датасета")
    name: str = Field(description="Название датасета")
    description: str = Field(description="Описание датасета")
    
    classes_count: int = Field(..., ge=1)
    classes_names: List[str] = Field(..., min_length=1)
    classes_to_idx: Dict[str, int] = Field(description="Словарь, где ключ название класса, а значение его индекс")

    type: Literal["image", "text", "tabular", "other"] = Field(description="Тип данных")
    task: Literal["classification", "regression", "detection", "segmentation", "other"] = Field(description="Название задачи")
    default_version_id: str = Field(description="ID версим являющейся стандартным значением")
    versions: List[Version] = Field(min_length=1, description="Список версий")

    created_at: datetime = Field(default_factory=datetime.now, frozen=True, description="Время создания датасета")
    updated_at: datetime = Field(default_factory=datetime.now, frozen=True, description="Время Изменения датасета")

    model_config = {
        "validate_assignment": True
    }

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        object.__setattr__(self, "updated_at", datetime.now())
    
    @model_validator(mode="after")
    def validate_default_version(self):

        version_ids = [v.id for v in self.versions]

        if self.default_version_id not in version_ids:
            raise ValueError(
                "Стандартная версия должна существовать в списке версий"
            )

        return self
    
    def get_datasets_response(self) -> DatasetResponse:
        versions = []
        for version in self.versions:
            versions.append(version.get_version_response())
        return DatasetResponse(
            **self.model_dump(exclude={'versions'}),
            versions=versions
        )