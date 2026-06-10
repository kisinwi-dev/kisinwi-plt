from typing import Dict, List
from pydantic import BaseModel, Field

class LeakageSummary(BaseModel):
    """Количество общих изображений (по хешу) между сплитами"""
    train_val: int = Field(0, ge=0, description="Общие изображения train и val")
    train_test: int = Field(0, ge=0, description="Общие изображения train и test")
    val_test: int = Field(0, ge=0, description="Общие изображения val и test")

class IntegritySummary(BaseModel):
    """Краткая сводка целостности данных версии"""
    duplicates_count: int = Field(0, ge=0, description="Количество лишних копий файлов внутри сплитов")
    leakage: LeakageSummary = Field(default_factory=LeakageSummary, description="Пересечения изображений между сплитами")

class DuplicateGroup(BaseModel):
    """Группа одинаковых файлов внутри одного сплита"""
    hash: str = Field(description="SHA256 содержимого")
    split: str = Field(description="Сплит, в котором найдены дубликаты")
    files: List[str] = Field(description="Относительные пути одинаковых файлов")

class LeakageGroup(BaseModel):
    """Одно изображение, встречающееся в нескольких сплитах"""
    hash: str = Field(description="SHA256 содержимого")
    files: Dict[str, List[str]] = Field(description="Словарь: сплит -> файлы с этим содержимым")

class IntegrityReportResponse(BaseModel):
    """Детальный отчёт о дубликатах и утечках между сплитами"""
    dataset_id: str = Field(description="ID датасета")
    version_id: str = Field(description="ID версии")
    summary: IntegritySummary = Field(description="Краткая сводка")
    duplicates: List[DuplicateGroup] = Field(default_factory=list, description="Группы дубликатов внутри сплитов")
    leakage: Dict[str, List[LeakageGroup]] = Field(
        default_factory=dict,
        description="Утечки между парами сплитов (ключи: train_val, train_test, val_test)"
    )
