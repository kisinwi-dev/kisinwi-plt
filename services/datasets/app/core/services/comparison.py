import math
from collections import Counter
from typing import Dict, Optional, Set, Tuple

from app.api.schemas.dataset import Version
from app.api.schemas.comparison import (
    ValueDelta, DriftLevel, ClassChanges, SplitDriftInfo,
    CountsComparisonResponse, DistributionComparisonResponse,
    BalanceComparisonResponse, SizeStatsComparisonResponse,
    FilesDiffResponse
)

# Замена нулевых пропорций при расчёте PSI (защита ln(0) и деления на ноль)
EPSILON = 1e-4

# Пороги интерпретации drift-метрик: ниже первого — none, между — moderate, выше второго — significant.
# Для PSI границы 0.1/0.25 — отраслевой стандарт; для JSD (log base 2, диапазон 0-1)
# общепринятого стандарта нет, используются те же границы как разумная конвенция.
DRIFT_THRESHOLDS: Tuple[float, float] = (0.1, 0.25)

def make_delta(from_value: float, to_value: float) -> ValueDelta:
    """Дельта значения между версиями; percent_change=None при нулевой базе"""
    delta = to_value - from_value
    percent_change = (delta / from_value * 100) if from_value != 0 else None
    return ValueDelta(
        from_value=from_value,
        to_value=to_value,
        delta=delta,
        percent_change=percent_change
    )

def _to_proportions(counts: Dict[str, int], keys: Set[str]) -> Optional[Dict[str, float]]:
    total = sum(counts.values())
    if total <= 0:
        return None
    return {k: counts.get(k, 0) / total for k in keys}

def js_divergence(p: Dict[str, int], q: Dict[str, int]) -> Optional[float]:
    """
    Jensen-Shannon divergence распределений классов (log base 2, диапазон 0-1).

    Args:
        * p, q — словари {class_name: count}

    Returns:
        None, если одно из распределений пустое
    """
    keys = set(p) | set(q)
    p_prop = _to_proportions(p, keys)
    q_prop = _to_proportions(q, keys)
    if p_prop is None or q_prop is None:
        return None

    def _kl_to_mixture(a: Dict[str, float], m: Dict[str, float]) -> float:
        # слагаемые с нулевой пропорцией равны 0 (lim x*log x = 0)
        return sum(
            a[k] * math.log2(a[k] / m[k])
            for k in keys if a[k] > 0
        )

    m = {k: (p_prop[k] + q_prop[k]) / 2 for k in keys}
    return (_kl_to_mixture(p_prop, m) + _kl_to_mixture(q_prop, m)) / 2

def psi(p: Dict[str, int], q: Dict[str, int]) -> Optional[float]:
    """
    Population Stability Index распределений классов.

    Args:
        * p, q — словари {class_name: count}

    Returns:
        None, если одно из распределений пустое
    """
    keys = set(p) | set(q)
    p_prop = _to_proportions(p, keys)
    q_prop = _to_proportions(q, keys)
    if p_prop is None or q_prop is None:
        return None

    result = 0.0
    for k in keys:
        p_k = max(p_prop[k], EPSILON)
        q_k = max(q_prop[k], EPSILON)
        result += (q_k - p_k) * math.log(q_k / p_k)
    return result

def drift_level(value: Optional[float]) -> Optional[DriftLevel]:
    """Интерпретация drift-метрики по порогам DRIFT_THRESHOLDS"""
    if value is None:
        return None
    low, high = DRIFT_THRESHOLDS
    if value < low:
        return DriftLevel.NONE
    if value <= high:
        return DriftLevel.MODERATE
    return DriftLevel.SIGNIFICANT

def _split_names_union(from_v: Version, to_v: Version) -> list[str]:
    names = {st.value for st in from_v.splits} | {st.value for st in to_v.splits}
    return sorted(names)

def _split_class_counts(version: Version) -> Dict[str, Dict[str, int]]:
    """{split_name: {class_name: count}} для всех сплитов версии"""
    return {st.value: split.get_class_counts() for st, split in version.splits.items()}

def compare_counts(
    dataset_id: str,
    from_v: Version,
    to_v: Version
) -> CountsComparisonResponse:
    """Сравнение количества изображений по сплитам и классам"""
    from_counts = _split_class_counts(from_v)
    to_counts = _split_class_counts(to_v)

    per_split = {}
    per_class = {}
    for split_name in _split_names_union(from_v, to_v):
        f_classes = from_counts.get(split_name, {})
        t_classes = to_counts.get(split_name, {})
        per_split[split_name] = make_delta(sum(f_classes.values()), sum(t_classes.values()))
        per_class[split_name] = {
            class_name: make_delta(f_classes.get(class_name, 0), t_classes.get(class_name, 0))
            for class_name in sorted(set(f_classes) | set(t_classes))
        }

    return CountsComparisonResponse(
        dataset_id=dataset_id,
        from_version_id=from_v.id,
        to_version_id=to_v.id,
        num_samples=make_delta(from_v.num_samples or 0, to_v.num_samples or 0),
        added_splits=sorted(set(to_counts) - set(from_counts)),
        removed_splits=sorted(set(from_counts) - set(to_counts)),
        per_split=per_split,
        per_class=per_class
    )

def compare_distribution(
    dataset_id: str,
    from_v: Version,
    to_v: Version
) -> DistributionComparisonResponse:
    """Сравнение распределений классов: изменения состава и drift-метрики"""
    from_counts = _split_class_counts(from_v)
    to_counts = _split_class_counts(to_v)

    class_changes = {}
    drift = {}
    for split_name in _split_names_union(from_v, to_v):
        f_classes = from_counts.get(split_name, {})
        t_classes = to_counts.get(split_name, {})

        class_changes[split_name] = ClassChanges(
            added_classes=sorted(set(t_classes) - set(f_classes)),
            removed_classes=sorted(set(f_classes) - set(t_classes)),
            common_classes=sorted(set(f_classes) & set(t_classes))
        )

        js_value = js_divergence(f_classes, t_classes)
        psi_value = psi(f_classes, t_classes)
        drift[split_name] = SplitDriftInfo(
            js_divergence=js_value,
            js_level=drift_level(js_value),
            psi=psi_value,
            psi_level=drift_level(psi_value)
        )

    return DistributionComparisonResponse(
        dataset_id=dataset_id,
        from_version_id=from_v.id,
        to_version_id=to_v.id,
        class_changes=class_changes,
        drift=drift
    )

def compare_balance(
    dataset_id: str,
    from_v: Version,
    to_v: Version
) -> BalanceComparisonResponse:
    """Сравнение коэффициентов баланса классов"""
    from_splits = {st.value: split for st, split in from_v.splits.items()}
    to_splits = {st.value: split for st, split in to_v.splits.items()}

    per_split = {}
    for split_name in _split_names_union(from_v, to_v):
        f_split = from_splits.get(split_name)
        t_split = to_splits.get(split_name)
        per_split[split_name] = make_delta(
            f_split.get_balance_ratio() if f_split else 0.0,
            t_split.get_balance_ratio() if t_split else 0.0
        )

    return BalanceComparisonResponse(
        dataset_id=dataset_id,
        from_version_id=from_v.id,
        to_version_id=to_v.id,
        overall_balance=make_delta(from_v._get_overall_balance(), to_v._get_overall_balance()),
        per_split=per_split
    )

def _split_size_counts(version: Version) -> Dict[str, Counter]:
    """{split_name: Counter{'WxH': count}} — полный счётчик размеров по сплитам"""
    result = {}
    for st, split in version.splits.items():
        size_counter = Counter()
        for cd in split.class_distribution:
            size_counter.update(cd.image_size_count)
        result[st.value] = size_counter
    return result

def compare_size_stats(
    dataset_id: str,
    from_v: Version,
    to_v: Version
) -> SizeStatsComparisonResponse:
    """Сравнение форматов и размеров изображений"""
    format_keys = sorted(set(from_v.image_format_stats) | set(to_v.image_format_stats))
    image_format_stats = {
        ext: make_delta(from_v.image_format_stats.get(ext, 0), to_v.image_format_stats.get(ext, 0))
        for ext in format_keys
    }

    from_sizes = _split_size_counts(from_v)
    to_sizes = _split_size_counts(to_v)
    size_counts_per_split = {}
    for split_name in _split_names_union(from_v, to_v):
        f_sizes = from_sizes.get(split_name, Counter())
        t_sizes = to_sizes.get(split_name, Counter())
        size_counts_per_split[split_name] = {
            size: make_delta(f_sizes.get(size, 0), t_sizes.get(size, 0))
            for size in sorted(set(f_sizes) | set(t_sizes))
        }

    return SizeStatsComparisonResponse(
        dataset_id=dataset_id,
        from_version_id=from_v.id,
        to_version_id=to_v.id,
        image_format_stats=image_format_stats,
        size_counts_per_split=size_counts_per_split
    )

def build_files_diff(
    dataset_id: str,
    from_version_id: str,
    to_version_id: str,
    from_files: Set[str],
    to_files: Set[str]
) -> FilesDiffResponse:
    """По-файловый diff по относительным путям (split/class/filename)"""
    added = sorted(to_files - from_files)
    removed = sorted(from_files - to_files)
    return FilesDiffResponse(
        dataset_id=dataset_id,
        from_version_id=from_version_id,
        to_version_id=to_version_id,
        added_count=len(added),
        removed_count=len(removed),
        common_count=len(from_files & to_files),
        added=added,
        removed=removed
    )
