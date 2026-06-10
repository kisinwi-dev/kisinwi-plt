import shutil
from pathlib import Path

import pytest
from PIL import Image

from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.exception.version import VersionValidationError
from app.core.filesystem import FileSystemManager
from app.core.services.validation.image_classification import (
    dataset_validation_and_create_metadata
)

CLASSES = ("cat", "dog")
SPLITS = ("train", "val", "test")


def _make_image(path: Path, color: tuple, mode: str = "RGB", size=(8, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)


def _build_valid_dataset(root: Path) -> None:
    """train/val/test x cat/dog, по одному уникальному png в каждом классе"""
    shade = 0
    for split in SPLITS:
        for cls in CLASSES:
            shade += 20
            _make_image(root / split / cls / f"img_{shade}.png", (shade, 0, 0))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "my_data"
    root.mkdir()
    _build_valid_dataset(root)
    return root


@pytest.fixture
def new_dataset() -> NewDataset:
    return NewDataset(
        name="test dataset",
        description="dataset for tests",
        type="image",
        task="classification",
        version=NewVersion(
            id_data="my_data",
            name="v1",
            description="first version",
            sources=[]
        )
    )


def _validate(data_root: Path, new_dataset: NewDataset):
    fsm = FileSystemManager(root=data_root)
    return dataset_validation_and_create_metadata(fsm, new_dataset)


# ================ happy path ======================

def test_valid_dataset_metadata_and_hashes(data_root, new_dataset):
    dsm, hashes = _validate(data_root, new_dataset)

    assert set(dsm.classes_names) == set(CLASSES)
    version = dsm.versions[0]
    assert version.num_samples == 6
    # формат суммируется по всем классам, а не затирается последним (регресс на баг)
    assert version.image_format_stats == {"png": 6}
    assert version.color_mode_stats == {"RGB": 6}
    assert version.integrity is not None
    assert version.integrity.duplicates_count == 0
    assert version.integrity.leakage.train_test == 0
    assert len(hashes) == 6
    assert all("/" in path for path in hashes)


def test_color_mode_stats_grayscale(data_root, new_dataset):
    _make_image(data_root / "train" / "cat" / "gray.png", 128, mode="L")

    dsm, _ = _validate(data_root, new_dataset)

    assert dsm.versions[0].color_mode_stats == {"RGB": 6, "L": 1}


# ================ битые изображения ======================

def test_broken_image_rejected(data_root, new_dataset):
    broken = data_root / "train" / "cat" / "broken.jpg"
    broken.write_bytes(b"this is not an image at all")

    with pytest.raises(VersionValidationError) as exc:
        _validate(data_root, new_dataset)

    assert "train/cat/broken.jpg" in str(exc.value.detail)


# ================ дубликаты и leakage ======================

def test_train_test_leakage_detected(data_root, new_dataset):
    src = next((data_root / "train" / "cat").glob("*.png"))
    shutil.copy(src, data_root / "test" / "cat" / "leaked_copy.png")

    dsm, _ = _validate(data_root, new_dataset)

    integrity = dsm.versions[0].integrity
    assert integrity.leakage.train_test == 1
    assert integrity.leakage.train_val == 0


def test_duplicates_inside_split_detected(data_root, new_dataset):
    src = next((data_root / "train" / "cat").glob("*.png"))
    shutil.copy(src, data_root / "train" / "cat" / "dup_copy.png")

    dsm, _ = _validate(data_root, new_dataset)

    assert dsm.versions[0].integrity.duplicates_count == 1


# ================ служебные файлы ======================

def test_junk_files_ignored(data_root, new_dataset):
    (data_root / "train" / "cat" / ".DS_Store").write_bytes(b"junk")
    (data_root / "val" / "dog" / "Thumbs.db").write_bytes(b"junk")

    dsm, hashes = _validate(data_root, new_dataset)

    assert dsm.versions[0].num_samples == 6
    assert len(hashes) == 6
