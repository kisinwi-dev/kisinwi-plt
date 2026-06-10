import math
import pytest

from app.api.schemas.comparison import DriftLevel
from app.api.schemas.dataset import Version
from app.core.exception.version import VersionComparisonError, VersionNotFoundError
from app.core.services import DatasetManager
from app.core.services.comparison import (
    make_delta, js_divergence, psi, drift_level,
    compare_counts, compare_distribution, compare_balance,
    compare_size_stats, build_files_diff
)


# ================ make_delta ======================

def test_make_delta():
    delta = make_delta(80, 100)
    assert delta.from_value == 80
    assert delta.to_value == 100
    assert delta.delta == 20
    assert delta.percent_change == 25.0


def test_make_delta_zero_base():
    delta = make_delta(0, 50)
    assert delta.delta == 50
    assert delta.percent_change is None


# ================ js_divergence ======================

def test_js_divergence_identical():
    counts = {"cat": 80, "dog": 100}
    assert js_divergence(counts, counts) == pytest.approx(0.0)


def test_js_divergence_disjoint():
    assert js_divergence({"cat": 100}, {"dog": 100}) == pytest.approx(1.0)


def test_js_divergence_empty():
    assert js_divergence({}, {"cat": 100}) is None
    assert js_divergence({"cat": 100}, {}) is None


def test_js_divergence_in_range():
    value = js_divergence({"cat": 80, "dog": 100}, {"cat": 100, "dog": 50, "bird": 50})
    assert 0.0 < value < 1.0


# ================ psi ======================

def test_psi_identical():
    counts = {"cat": 80, "dog": 100}
    assert psi(counts, counts) == pytest.approx(0.0)


def test_psi_one_sided_class_is_finite():
    value = psi({"cat": 50, "dog": 50}, {"cat": 100})
    assert math.isfinite(value)
    assert value > 0.25  # исчезновение класса — significant drift


def test_psi_empty():
    assert psi({}, {"cat": 100}) is None
    assert psi({"cat": 100}, {}) is None


# ================ drift_level ======================

def test_drift_level():
    assert drift_level(None) is None
    assert drift_level(0.05) == DriftLevel.NONE
    assert drift_level(0.15) == DriftLevel.MODERATE
    assert drift_level(0.25) == DriftLevel.MODERATE
    assert drift_level(0.3) == DriftLevel.SIGNIFICANT


# ================ compare_counts ======================

def test_compare_counts(from_version: Version, to_version: Version):
    result = compare_counts("ds1", from_version, to_version)

    assert result.num_samples.from_value == 220
    assert result.num_samples.to_value == 220
    assert result.added_splits == ["test"]
    assert result.removed_splits == ["val"]

    train = result.per_split["train"]
    assert train.from_value == 180
    assert train.to_value == 200

    # val отсутствует в to_version — считается нулём
    val = result.per_split["val"]
    assert val.to_value == 0
    assert val.delta == -40

    cat = result.per_class["train"]["cat"]
    assert cat.delta == 20
    assert cat.percent_change == 25.0

    # bird появился — нулевая база, percent_change нет
    bird = result.per_class["train"]["bird"]
    assert bird.from_value == 0
    assert bird.percent_change is None


def test_compare_counts_empty_version(from_version: Version, empty_version: Version):
    result = compare_counts("ds1", from_version, empty_version)
    assert result.removed_splits == ["train", "val"]
    assert result.added_splits == []
    assert result.num_samples.to_value == 0


# ================ compare_distribution ======================

def test_compare_distribution(from_version: Version, to_version: Version):
    result = compare_distribution("ds1", from_version, to_version)

    train_changes = result.class_changes["train"]
    assert train_changes.added_classes == ["bird"]
    assert train_changes.removed_classes == []
    assert train_changes.common_classes == ["cat", "dog"]

    train_drift = result.drift["train"]
    assert 0.0 < train_drift.js_divergence < 1.0
    assert train_drift.js_level is not None
    assert train_drift.psi > 0
    assert train_drift.psi_level is not None

    # val есть только в from_version — drift не считается
    val_drift = result.drift["val"]
    assert val_drift.js_divergence is None
    assert val_drift.psi is None
    assert val_drift.js_level is None
    assert val_drift.psi_level is None


# ================ compare_balance ======================

def test_compare_balance(from_version: Version, to_version: Version):
    result = compare_balance("ds1", from_version, to_version)

    train = result.per_split["train"]
    assert train.from_value == pytest.approx(80 / 100)
    assert train.to_value == pytest.approx(50 / 100)

    # val отсутствует в to_version — баланс 0
    assert result.per_split["val"].to_value == 0.0

    # from: cat 100, dog 120 -> 100/120; to: cat 110, dog 60, bird 50 -> 50/110
    assert result.overall_balance.from_value == pytest.approx(100 / 120)
    assert result.overall_balance.to_value == pytest.approx(50 / 110)


# ================ compare_size_stats ======================

def test_compare_size_stats(from_version: Version, to_version: Version):
    result = compare_size_stats("ds1", from_version, to_version)

    assert result.image_format_stats["jpg"].delta == 20
    # png исчез — присутствует в union с нулём
    assert result.image_format_stats["png"].to_value == 0

    train_sizes = result.size_counts_per_split["train"]
    assert train_sizes["224x224"].from_value == 160
    assert train_sizes["224x224"].to_value == 150
    assert train_sizes["512x512"].from_value == 0
    assert train_sizes["512x512"].to_value == 50


# ================ build_files_diff ======================

def test_build_files_diff():
    result = build_files_diff(
        "ds1", "v1", "v2",
        from_files={"train/cat/a.jpg", "train/cat/b.jpg", "val/dog/c.jpg"},
        to_files={"train/cat/a.jpg", "train/bird/d.jpg"},
    )
    assert result.added == ["train/bird/d.jpg"]
    assert result.removed == ["train/cat/b.jpg", "val/dog/c.jpg"]
    assert result.added_count == 1
    assert result.removed_count == 2
    assert result.common_count == 1


# ================ DatasetManager.compare_* ======================

def test_manager_compare_same_version_raises(manager: DatasetManager):
    with pytest.raises(VersionComparisonError):
        manager.compare_version_counts("ds1", "v1", "v1")


def test_manager_compare_unknown_version_raises(manager: DatasetManager):
    with pytest.raises(VersionNotFoundError):
        manager.compare_version_counts("ds1", "v1", "v999")


def test_manager_compare_version_files(manager: DatasetManager):
    result = manager.compare_version_files("ds1", "v1", "v2")
    assert result.added == ["train/bird/img4.jpg"]
    assert result.removed == ["train/cat/img2.jpg", "train/dog/img3.jpg"]
    assert result.common_count == 1


def test_manager_compare_versions_summary(manager: DatasetManager):
    result = manager.compare_versions("ds1", "v1", "v2")

    assert result.dataset_id == "ds1"
    assert result.from_version_id == "v1"
    assert result.to_version_id == "v2"
    assert result.counts.per_split["train"].delta == 20
    assert result.distribution.class_changes["train"].added_classes == ["bird"]
    assert result.balance.per_split["train"].from_value == pytest.approx(0.8)
    assert result.size_stats.image_format_stats["jpg"].delta == 20
    # в сводке только счётчики файлового diff
    assert result.files.added_count == 1
    assert result.files.removed_count == 2
    assert result.files.common_count == 1
