from typing import List, Optional

from app.api.schemas import (
    ModelMetricData,
    ModelMetrics,
    MetricSummary,
    OverfittingGap,
    ModelMetricsSummary,
    MetricComparison,
    ModelComparisonEntry,
    ModelsCompareResponse,
    Split,
)

SPLITS = ("train", "val", "test")

# Метрики, у которых меньшее значение лучше (по подстроке в названии)
LOWER_IS_BETTER_TOKENS = ("loss", "error", "err", "mae", "mse", "rmse")

def is_higher_better(metric_name: str) -> bool:
    """Направление метрики: loss/error-подобные — чем меньше, тем лучше, остальные — чем больше"""
    name = metric_name.lower()
    return not any(token in name for token in LOWER_IS_BETTER_TOKENS)

def _numeric(values: list) -> List[float]:
    """Только числовые значения (values объявлены как List[Any])"""
    return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]

def summarize_metric(metric: ModelMetricData) -> Optional[MetricSummary]:
    """Сводка по метрике: final/best/min/max и эпоха лучшего значения; None без числовых значений"""
    values = _numeric(metric.values)
    if not values:
        return None

    higher_is_better = is_higher_better(metric.name)
    best_value = max(values) if higher_is_better else min(values)

    return MetricSummary(
        name=metric.name,
        final_value=values[-1],
        best_value=best_value,
        best_epoch=values.index(best_value) + 1,
        min_value=min(values),
        max_value=max(values),
        epochs=len(values),
        higher_is_better=higher_is_better,
    )

def _overfitting_gaps(
    train: List[MetricSummary],
    val: List[MetricSummary],
) -> List[OverfittingGap]:
    """Разрывы train/val на последней эпохе; gap > 0 — на валидации хуже"""
    val_by_name = {summary.name: summary for summary in val}
    gaps = []
    for train_summary in train:
        val_summary = val_by_name.get(train_summary.name)
        if val_summary is None:
            continue
        if train_summary.higher_is_better:
            gap = train_summary.final_value - val_summary.final_value
        else:
            gap = val_summary.final_value - train_summary.final_value
        gaps.append(OverfittingGap(
            metric=train_summary.name,
            train_final=train_summary.final_value,
            val_final=val_summary.final_value,
            gap=gap,
        ))
    return gaps

def compute_model_summary(metrics: ModelMetrics) -> ModelMetricsSummary:
    """Сводная статистика модели по выборкам + разрывы train/val"""
    summaries = {
        split: [
            summary
            for metric in getattr(metrics, split)
            if (summary := summarize_metric(metric)) is not None
        ]
        for split in SPLITS
    }
    return ModelMetricsSummary(
        model_id=metrics.model_id,
        overfitting=_overfitting_gaps(summaries["train"], summaries["val"]),
        **summaries,
    )

def compare_models(
    all_metrics: List[ModelMetrics],
    requested_ids: List[str],
    split: Split,
    metric_filter: Optional[List[str]] = None,
) -> ModelsCompareResponse:
    """Сравнение моделей по метрикам выбранной выборки.

    Модели без сохранённых метрик попадают в missing; лидер по метрике
    определяется лучшим best_value с учётом направления метрики.
    """
    found_ids = {m.model_id for m in all_metrics}
    missing = [model_id for model_id in requested_ids if model_id not in found_ids]

    # Сводки метрик выбранной выборки по каждой модели
    summaries_by_model = {
        model.model_id: {
            summary.name: summary
            for metric in getattr(model, split)
            if (summary := summarize_metric(metric)) is not None
        }
        for model in all_metrics
    }

    metric_names = sorted({
        name
        for summaries in summaries_by_model.values()
        for name in summaries
    })
    if metric_filter is not None:
        allowed = set(metric_filter)
        metric_names = [name for name in metric_names if name in allowed]

    comparisons = []
    for name in metric_names:
        higher_is_better = is_higher_better(name)
        participants = [
            (model_id, summaries[name])
            for model_id, summaries in summaries_by_model.items()
            if name in summaries
        ]

        best = (max if higher_is_better else min)(
            participants, key=lambda pair: pair[1].best_value
        )
        best_model_id, best_summary = best

        entries = [
            ModelComparisonEntry(
                model_id=model_id,
                final_value=summary.final_value,
                best_value=summary.best_value,
                best_epoch=summary.best_epoch,
                epochs=summary.epochs,
                delta_best=abs(summary.best_value - best_summary.best_value),
            )
            for model_id, summary in participants
        ]
        entries.sort(key=lambda entry: entry.delta_best)

        comparisons.append(MetricComparison(
            metric=name,
            higher_is_better=higher_is_better,
            best_model_id=best_model_id,
            models=entries,
        ))

    return ModelsCompareResponse(split=split, metrics=comparisons, missing=missing)
