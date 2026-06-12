from typing import List, NamedTuple, Optional

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
        checkpoint=metrics.checkpoint,
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
    определяется лучшим weights_value — значением на эпохе сохранённых весов
    (чекпоинт по early-stop-метрике), с фолбэком на final_value у моделей
    без записанного чекпоинта, — с учётом направления метрики.
    """
    found_ids = {m.model_id for m in all_metrics}
    missing = [model_id for model_id in requested_ids if model_id not in found_ids]

    # Сводки и числовые значения метрик выбранной выборки по каждой модели:
    # сырые values нужны для среза на эпохе чекпоинта.
    stats_by_model = {
        model.model_id: {
            summary.name: (summary, _numeric(metric.values))
            for metric in getattr(model, split)
            if (summary := summarize_metric(metric)) is not None
        }
        for model in all_metrics
    }
    checkpoint_by_model = {m.model_id: m.checkpoint for m in all_metrics}

    metric_names = sorted({
        name
        for stats in stats_by_model.values()
        for name in stats
    })
    if metric_filter is not None:
        allowed = set(metric_filter)
        metric_names = [name for name in metric_names if name in allowed]

    class _Participant(NamedTuple):
        model_id: str
        summary: MetricSummary
        checkpoint_epoch: Optional[int]
        checkpoint_value: Optional[float]
        weights_value: float

    comparisons = []
    for name in metric_names:
        higher_is_better = is_higher_better(name)
        participants = []
        for model_id, stats in stats_by_model.items():
            if name not in stats:
                continue
            summary, values = stats[name]
            ckpt = checkpoint_by_model.get(model_id)
            checkpoint_value = (
                values[ckpt.epoch - 1]
                if ckpt is not None and 1 <= ckpt.epoch <= len(values)
                else None
            )
            participants.append(_Participant(
                model_id=model_id,
                summary=summary,
                checkpoint_epoch=ckpt.epoch if ckpt is not None else None,
                # На одноточечном test чекпоинт-среза нет (values уже измерены
                # на сохранённых весах) — фолбэк на final_value корректен.
                checkpoint_value=checkpoint_value,
                weights_value=checkpoint_value if checkpoint_value is not None else summary.final_value,
            ))

        best = (max if higher_is_better else min)(
            participants, key=lambda p: p.weights_value
        )

        entries = [
            ModelComparisonEntry(
                model_id=p.model_id,
                final_value=p.summary.final_value,
                best_value=p.summary.best_value,
                best_epoch=p.summary.best_epoch,
                epochs=p.summary.epochs,
                weights_value=p.weights_value,
                checkpoint_epoch=p.checkpoint_epoch,
                checkpoint_value=p.checkpoint_value,
                delta_best=abs(p.weights_value - best.weights_value),
            )
            for p in participants
        ]
        entries.sort(key=lambda entry: entry.delta_best)

        comparisons.append(MetricComparison(
            metric=name,
            higher_is_better=higher_is_better,
            best_model_id=best.model_id,
            models=entries,
        ))

    return ModelsCompareResponse(split=split, metrics=comparisons, missing=missing)
