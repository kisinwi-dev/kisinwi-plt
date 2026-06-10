import pytest
from pathlib import Path

from app.api.schemas.dataset import DatasetMetadata, Version
from app.api.schemas.splits import ClassDistribution, Split, SplitType
from app.core.filesystem import FileSystemManager
from app.core.services import DatasetManager


@pytest.fixture()
def from_version() -> Version:
    """Базовая версия: train (cat 80, dog 100) + val (cat 20, dog 20)"""
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
        id="v1",
        name="v1",
        description="base version",
        sources=[],
        size_bytes=1000,
        image_format_stats={"jpg": 200, "png": 20},
        splits={SplitType.TRAIN: train, SplitType.VAL: val},
    )


@pytest.fixture()
def to_version() -> Version:
    """Сравниваемая версия: train (cat 100, dog 50, bird 50) + test; val удалён"""
    train = Split(class_distribution=[
        ClassDistribution(
            class_name="cat", class_id=0, count=100,
            image_size_count={"224x224": 100}
        ),
        ClassDistribution(
            class_name="dog", class_id=1, count=50,
            image_size_count={"512x512": 50}
        ),
        ClassDistribution(
            class_name="bird", class_id=2, count=50,
            image_size_count={"224x224": 50}
        ),
    ])
    test = Split(class_distribution=[
        ClassDistribution(
            class_name="cat", class_id=0, count=10,
            image_size_count={"224x224": 10}
        ),
        ClassDistribution(
            class_name="dog", class_id=1, count=10,
            image_size_count={"224x224": 10}
        ),
    ])
    return Version(
        id="v2",
        name="v2",
        description="updated version",
        sources=[],
        size_bytes=2000,
        image_format_stats={"jpg": 220},
        splits={SplitType.TRAIN: train, SplitType.TEST: test},
    )


@pytest.fixture()
def empty_version() -> Version:
    """Версия без сплитов"""
    return Version(
        id="v0",
        name="v0",
        description="empty version",
        sources=[],
        size_bytes=0,
        splits={},
    )


@pytest.fixture()
def manager(tmp_path: Path, from_version: Version, to_version: Version) -> DatasetManager:
    """DatasetManager на tmp_path с датасетом ds1 (версии v1 и v2 на диске)"""
    dataset_path = tmp_path / "ds1"
    dataset_path.mkdir()

    dsm = DatasetMetadata(
        id="ds1",
        name="ds1",
        description="test dataset",
        classes_count=3,
        classes_names=["cat", "dog", "bird"],
        classes_to_idx={"cat": 0, "dog": 1, "bird": 2},
        type="image",
        task="classification",
        default_version_id="v1",
        versions=[from_version, to_version],
    )
    (dataset_path / "metadata_ds.json").write_text(
        dsm.model_dump_json(indent=2), encoding="utf-8"
    )

    files_per_version = {
        "v1": ["train/cat/img1.jpg", "train/cat/img2.jpg", "train/dog/img3.jpg"],
        "v2": ["train/cat/img1.jpg", "train/bird/img4.jpg"],
    }
    for version_id, files in files_per_version.items():
        for rel_path in files:
            file_path = dataset_path / version_id / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("img")

    dm = DatasetManager.__new__(DatasetManager)
    dm._fsm = FileSystemManager(root=tmp_path)
    dm._temp_path = (tmp_path / "temp").resolve()
    return dm
