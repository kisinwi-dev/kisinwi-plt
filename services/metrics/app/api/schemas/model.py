from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from typing import List, Any, Literal, Optional

Split = Literal["train", "val", "test"]

TrainingStatus = Literal["in_progress", "completed", "failed", "cancelled"]

class ModelMetricData(BaseModel):
    """Схема метрик"""
    name: str = Field(..., description="Название метрики", examples=["loss"])
    split: Optional[Split] = Field(
        None,
        description="Выборка (train/val/test); если не задана — выводится из префикса названия",
        examples=["train"],
    )
    values: List[Any] = Field(
        default_factory=list,
        description="Значения метрик",
        examples=[[0.91, 0.55, 0.42]],
    )
    timestamps: List[datetime] = Field(
        default_factory=list,
        description="Метки времени записи значений (UTC), параллельны values; "
                    "у моделей, обученных до ввода timestamps, массив может быть "
                    "короче values — выравнивать с конца",
        examples=[["2026-06-11T10:00:00Z", "2026-06-11T10:01:30Z", "2026-06-11T10:03:00Z"]],
    )

class ModelMetricAdd(BaseModel):
    """Схема на добавление метрики"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    metric: ModelMetricData = Field(..., description="Метрика")
    timestamp: Optional[datetime] = Field(
        None,
        description="Метка времени записи (UTC); если не задана — проставляется серверное время",
        examples=["2026-06-11T10:00:00Z"],
    )

class ModelMetricAdds(BaseModel):
    """Схема на добавление нескольких метрик"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    metrics: List[ModelMetricData] = Field(
        default_factory=list,
        description="Список метрик",
        examples=[[
            {"name": "train_loss", "values": [0.91]},
            {"name": "val_loss", "values": [1.02]},
            {"name": "val_accuracy", "values": [0.63]},
        ]],
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Общая метка времени записи для всех метрик батча (UTC); "
                    "если не задана — проставляется серверное время",
        examples=["2026-06-11T10:00:00Z"],
    )

class ModelMetrics(BaseModel):
    """ID модели и её метрики, разбитые по выборкам"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    status: Optional[TrainingStatus] = Field(
        None,
        description="Статус обучения; None у моделей, обученных до ввода статусов",
        examples=["in_progress"],
    )
    train: List[ModelMetricData] = Field(default_factory=list, description="Метрики тренировочной выборки")
    val: List[ModelMetricData] = Field(default_factory=list, description="Метрики валидационной выборки")
    test: List[ModelMetricData] = Field(default_factory=list, description="Метрики тестовой выборки")

class ModelTrainingStatusUpdate(BaseModel):
    """Установка статуса обучения модели"""
    status: TrainingStatus = Field(
        ...,
        description="Статус обучения: in_progress ставится на старте (в т.ч. сбрасывает "
                    "финальный статус при переобучении), completed/failed/cancelled — финальные",
        examples=["completed"],
    )

class ModelTrainingStatusResponse(BaseModel):
    """Текущий статус обучения модели"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    status: TrainingStatus = Field(..., description="Статус обучения", examples=["completed"])

class PerClassMetrics(BaseModel):
    """Метрики качества по одному классу на тестовой выборке"""
    label: str = Field(..., description="Название класса", examples=["cat"])
    precision: float = Field(..., description="Precision класса", examples=[0.89])
    recall: float = Field(..., description="Recall класса", examples=[0.8])
    f1: float = Field(..., description="F1 класса", examples=[0.84])
    support: int = Field(..., description="Число примеров класса в тестовой выборке", examples=[10])

class ClassReportAdd(BaseModel):
    """Отчёт по классам на тестовой выборке: confusion matrix и per-class метрики"""
    labels: List[str] = Field(
        ...,
        description="Названия классов; порядок задаёт строки/столбцы confusion matrix",
        examples=[["cat", "dog"]],
    )
    confusion_matrix: List[List[int]] = Field(
        ...,
        description="Confusion matrix (без нормализации): строки — истинные классы, "
                    "столбцы — предсказанные, порядок как в labels",
        examples=[[[8, 2], [1, 9]]],
    )
    per_class: List[PerClassMetrics] = Field(
        ...,
        description="Метрики по каждому классу, порядок как в labels",
    )

    @model_validator(mode="after")
    def check_dimensions(self):
        """Квадратная матрица, согласованная с labels и per_class"""
        n = len(self.labels)
        if len(self.confusion_matrix) != n or any(len(row) != n for row in self.confusion_matrix):
            raise ValueError(f"confusion_matrix должна быть квадратной {n}x{n} (по числу labels)")
        if len(self.per_class) != n:
            raise ValueError("per_class должен содержать запись на каждый класс из labels")
        return self

class ClassReport(ClassReportAdd):
    """Отчёт по классам с ID модели"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])

class ModelMetricsBatchRequest(BaseModel):
    """Запрос метрик сразу нескольких моделей"""
    model_ids: List[str] = Field(
        default_factory=list,
        description="Список ID моделей",
        examples=[["model-42", "model-43"]],
    )

class StatusResponse(BaseModel):
    """Статус выполнения операции"""
    status: str = Field(..., description="Статус операции", examples=["ok"])

class ModelExistsResponse(BaseModel):
    """Признак наличия метрик модели"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    exists: bool = Field(..., description="Есть ли сохранённые метрики модели")

class ModelDeleteResponse(BaseModel):
    """Результат удаления метрик модели"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    deleted: bool = Field(..., description="Признак удаления метрик")

class MetricSummary(BaseModel):
    """Сводка по одной метрике в рамках выборки"""
    name: str = Field(..., description="Название метрики", examples=["loss"])
    final_value: float = Field(..., description="Значение на последней эпохе")
    best_value: float = Field(..., description="Лучшее значение с учётом направления метрики")
    best_epoch: int = Field(..., description="Эпоха лучшего значения (нумерация с 1)")
    min_value: float = Field(..., description="Минимальное значение за все эпохи")
    max_value: float = Field(..., description="Максимальное значение за все эпохи")
    epochs: int = Field(..., description="Число эпох с записанными значениями")
    higher_is_better: bool = Field(
        ...,
        description="Направление метрики: True — чем больше, тем лучше "
                    "(loss/error-подобные — чем меньше, тем лучше)",
    )

class OverfittingGap(BaseModel):
    """Разрыв train/val на последней эпохе; gap > 0 — на валидации хуже (признак переобучения)"""
    metric: str = Field(..., description="Название метрики", examples=["accuracy"])
    train_final: float = Field(..., description="Значение на train на последней эпохе")
    val_final: float = Field(..., description="Значение на val на последней эпохе")
    gap: float = Field(
        ...,
        description="Разрыв с учётом направления метрики: положительный — на валидации хуже",
    )

class ModelMetricsSummary(BaseModel):
    """Сводная статистика метрик модели по выборкам"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    train: List[MetricSummary] = Field(default_factory=list, description="Сводка по тренировочной выборке")
    val: List[MetricSummary] = Field(default_factory=list, description="Сводка по валидационной выборке")
    test: List[MetricSummary] = Field(default_factory=list, description="Сводка по тестовой выборке")
    overfitting: List[OverfittingGap] = Field(
        default_factory=list,
        description="Разрывы train/val по метрикам, присутствующим в обеих выборках",
    )

class ModelsCompareRequest(BaseModel):
    """Запрос сравнения нескольких моделей"""
    model_ids: List[str] = Field(
        ...,
        min_length=2,
        description="ID сравниваемых моделей (минимум 2)",
        examples=[["model-42", "model-43"]],
    )
    split: Split = Field("val", description="Выборка, по которой сравниваются модели")
    metrics: Optional[List[str]] = Field(
        None,
        description="Фильтр по названиям метрик; если не задан — сравниваются все",
        examples=[["accuracy", "loss"]],
    )

class ModelComparisonEntry(BaseModel):
    """Показатели одной модели по метрике"""
    model_id: str = Field(..., description="ID модели", examples=["model-42"])
    final_value: float = Field(..., description="Значение на последней эпохе")
    best_value: float = Field(..., description="Лучшее значение с учётом направления метрики")
    best_epoch: int = Field(..., description="Эпоха лучшего значения (нумерация с 1)")
    epochs: int = Field(..., description="Число эпох с записанными значениями")
    delta_best: float = Field(
        ...,
        description="Отставание best_value от лидера по этой метрике (0 у лидера)",
    )

class MetricComparison(BaseModel):
    """Сравнение моделей по одной метрике"""
    metric: str = Field(..., description="Название метрики", examples=["accuracy"])
    higher_is_better: bool = Field(..., description="Направление метрики")
    best_model_id: Optional[str] = Field(None, description="ID модели-лидера по лучшему значению")
    models: List[ModelComparisonEntry] = Field(
        default_factory=list,
        description="Показатели каждой модели, у которой есть эта метрика",
    )

class ModelsCompareResponse(BaseModel):
    """Результат сравнения моделей по метрикам выбранной выборки"""
    split: Split = Field(..., description="Выборка сравнения")
    metrics: List[MetricComparison] = Field(default_factory=list, description="Сравнение по каждой метрике")
    missing: List[str] = Field(
        default_factory=list,
        description="Запрошенные модели, у которых нет сохранённых метрик",
    )
