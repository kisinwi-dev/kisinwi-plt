from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class DriftLevel(str, Enum):
    NONE = "none"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"

class ValueDelta(BaseModel):
    """Изменение значения между двумя версиями"""
    from_value: float = Field(description="Значение в базовой версии")
    to_value: float = Field(description="Значение в сравниваемой версии")
    delta: float = Field(description="Разница (to - from)")
    percent_change: Optional[float] = Field(
        default=None,
        description="Изменение в процентах от базового значения (None, если базовое значение 0)"
    )

class ComparisonBase(BaseModel):
    """Общие поля всех ответов сравнения версий"""
    dataset_id: str = Field(description="ID датасета")
    from_version_id: str = Field(description="ID базовой версии")
    to_version_id: str = Field(description="ID сравниваемой версии")

class CountsComparisonResponse(ComparisonBase):
    """Сравнение количества изображений между версиями"""
    num_samples: ValueDelta = Field(description="Изменение общего количества изображений")
    added_splits: List[str] = Field(description="Сплиты, появившиеся в сравниваемой версии")
    removed_splits: List[str] = Field(description="Сплиты, отсутствующие в сравниваемой версии")
    per_split: Dict[str, ValueDelta] = Field(description="Изменение количества изображений по сплитам")
    per_class: Dict[str, Dict[str, ValueDelta]] = Field(
        description="Изменение количества изображений по классам в каждом сплите"
    )

class ClassChanges(BaseModel):
    """Изменения состава классов одного сплита"""
    added_classes: List[str] = Field(description="Классы, появившиеся в сравниваемой версии")
    removed_classes: List[str] = Field(description="Классы, отсутствующие в сравниваемой версии")
    common_classes: List[str] = Field(description="Классы, присутствующие в обеих версиях")

class SplitDriftInfo(BaseModel):
    """Drift-метрики распределения классов одного сплита"""
    js_divergence: Optional[float] = Field(
        default=None,
        description="Jensen-Shannon divergence (log base 2, диапазон 0-1; None при пустом сплите)"
    )
    js_level: Optional[DriftLevel] = Field(default=None, description="Интерпретация JS divergence")
    psi: Optional[float] = Field(
        default=None,
        description="Population Stability Index (None при пустом сплите)"
    )
    psi_level: Optional[DriftLevel] = Field(default=None, description="Интерпретация PSI")

class DistributionComparisonResponse(ComparisonBase):
    """Сравнение распределений классов между версиями"""
    class_changes: Dict[str, ClassChanges] = Field(description="Изменения состава классов по сплитам")
    drift: Dict[str, SplitDriftInfo] = Field(description="Drift-метрики распределения по сплитам")

class BalanceComparisonResponse(ComparisonBase):
    """Сравнение баланса классов между версиями"""
    overall_balance: ValueDelta = Field(description="Изменение общего коэффициента баланса")
    per_split: Dict[str, ValueDelta] = Field(description="Изменение коэффициента баланса по сплитам")

class SizeStatsComparisonResponse(ComparisonBase):
    """Сравнение размеров и форматов изображений между версиями"""
    image_format_stats: Dict[str, ValueDelta] = Field(
        description="Изменение количества изображений по форматам"
    )
    size_counts_per_split: Dict[str, Dict[str, ValueDelta]] = Field(
        description="Изменение количества изображений по размерам (WxH) в каждом сплите"
    )

class FilesDiffSummary(BaseModel):
    """Счётчики по-файлового diff"""
    added_count: int = Field(description="Количество добавленных файлов")
    removed_count: int = Field(description="Количество удалённых файлов")
    common_count: int = Field(description="Количество файлов, присутствующих в обеих версиях")

class FilesDiffResponse(ComparisonBase, FilesDiffSummary):
    """По-файловый diff между версиями (по относительным путям, без хешей)"""
    added: List[str] = Field(description="Добавленные файлы (пути вида train/cat/img.jpg)")
    removed: List[str] = Field(description="Удалённые файлы (пути вида train/cat/img.jpg)")

class VersionComparisonResponse(ComparisonBase):
    """Полная сводка сравнения двух версий датасета"""
    counts: CountsComparisonResponse = Field(description="Сравнение количества изображений")
    distribution: DistributionComparisonResponse = Field(description="Сравнение распределений классов")
    balance: BalanceComparisonResponse = Field(description="Сравнение баланса классов")
    size_stats: SizeStatsComparisonResponse = Field(description="Сравнение размеров и форматов")
    files: FilesDiffSummary = Field(description="Счётчики по-файлового diff (списки — в /compare/files)")
