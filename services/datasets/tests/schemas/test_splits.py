import pytest

from app.api.schemas.dataset import Version
from app.api.schemas.splits import ClassDistribution, Split, SplitType


@pytest.fixture()
def version() -> Version:
    """Версия с train/val сплитами, построенная программно"""
    train = Split(class_distribution=[
        ClassDistribution(
            class_name="cat", class_id=0, count=80,
            image_size_count={"224x224": 60, "300x300": 20}
        ),
        ClassDistribution(
            class_name="dog", class_id=1, count=100,
            image_size_count={"224x224": 100}
        ),
    ])
    val = Split(class_distribution=[
        ClassDistribution(
            class_name="cat", class_id=0, count=20,
            image_size_count={"224x224": 20}
        ),
        ClassDistribution(
            class_name="dog", class_id=1, count=20,
            image_size_count={"224x224": 15, "300x300": 5}
        ),
    ])
    return Version(
        id="v1.0",
        name="v1.0",
        description="test version",
        sources=[],
        size_bytes=1000,
        splits={SplitType.TRAIN: train, SplitType.VAL: val},
    )


@pytest.fixture()
def empty_version() -> Version:
    """Версия без сплитов"""
    return Version(
        id="v0.0",
        name="v0.0",
        description="empty version",
        sources=[],
        size_bytes=0,
        splits={},
    )


# ================ get_split_counts ======================

def test_split_counts(version: Version):
    counts = version.get_split_counts()
    assert counts.id == "v1.0"
    assert counts.counts_per_split == {"train": 180, "val": 40}
    assert sum(counts.counts_per_split.values()) == counts.num_samples


def test_split_counts_empty_version(empty_version: Version):
    counts = empty_version.get_split_counts()
    assert counts.num_samples == 0
    assert counts.counts_per_split == {}


# ================ get_split_balance ======================

def test_split_balance(version: Version):
    balance = version.get_split_balance()

    train = balance.splits["train"]
    assert train.total_samples == 180
    assert train.num_classes == 2
    assert train.balance_ratio == 80 / 100
    assert train.is_balanced  # 0.8 >= 0.7

    val = balance.splits["val"]
    assert val.balance_ratio == 1.0
    assert val.is_balanced

    # cat: 100, dog: 120 по всем сплитам
    assert balance.overall_balance == 100 / 120


def test_split_balance_empty_version(empty_version: Version):
    balance = empty_version.get_split_balance()
    assert balance.splits == {}
    assert balance.overall_balance == 0.0


# ================ get_class_distribution_response ======================

def test_class_distribution(version: Version):
    distribution = version.get_class_distribution_response()

    assert set(distribution.splits.keys()) == {"train", "val"}

    train = distribution.splits["train"]
    by_name = {item.class_name: item for item in train}
    assert set(by_name.keys()) == {"cat", "dog"}
    assert by_name["cat"].class_id == 0
    assert by_name["cat"].count == 80
    assert by_name["dog"].count == 100


def test_class_distribution_empty_version(empty_version: Version):
    distribution = empty_version.get_class_distribution_response()
    assert distribution.splits == {}


# ================ get_image_size_stats ======================

def test_image_size_stats(version: Version):
    stats = version.get_image_size_stats()

    train = stats.splits["train"]
    assert train.total_images == 180
    assert train.unique_sizes == 2
    assert train.most_common_size == "224x224"
    assert train.most_common_count == 160
    assert train.size_consistency == round(160 / 180, 2)
    assert train.top_10_sizes == {"224x224": 160, "300x300": 20}


def test_image_size_stats_empty_split():
    version = Version(
        id="v1.0",
        name="v1.0",
        description="version with empty split",
        sources=[],
        num_samples=0,
        size_bytes=0,
        splits={SplitType.TEST: Split()},
    )
    stats = version.get_image_size_stats()

    test = stats.splits["test"]
    assert test.unique_sizes == 0
    assert test.total_images == 0
    assert test.most_common_size is None
    assert test.most_common_count is None
    assert test.size_consistency is None
    assert test.top_10_sizes == {}
