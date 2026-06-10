import json
from pathlib import Path

import pytest
from PIL import Image

from app.api.schemas.dataset import DatasetMetadata, Version
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.api.schemas.splits import SplitType
from app.core.exception.dataset import MetadataSaveError
from app.core.exception.version import (
    VersionNotFoundError, IntegrityReportNotAvailableError
)
from app.core.services import DatasetManager


# ================ список датасетов: пагинация и поиск ======================

def test_list_datasets_response_total(manager: DatasetManager):
    datasets, total = manager.list_datasets_response()
    assert total == 1
    assert datasets[0].id == "ds1"


def test_list_datasets_response_search(manager: DatasetManager):
    _, total = manager.list_datasets_response(search="DS1")
    assert total == 1

    datasets, total = manager.list_datasets_response(search="nothing")
    assert total == 0
    assert datasets == []


def test_list_datasets_response_offset_beyond_total(manager: DatasetManager):
    datasets, total = manager.list_datasets_response(limit=10, offset=5)
    assert total == 1
    assert datasets == []


# ================ обновление метаданных ======================

def test_update_dataset_info_persists(manager: DatasetManager):
    updated = manager.update_dataset_info("ds1", name="renamed", description="new desc")
    assert updated.name == "renamed"
    assert updated.description == "new desc"

    reloaded = manager._get_dataset_info("ds1")
    assert reloaded.name == "renamed"
    assert reloaded.description == "new desc"


def test_update_dataset_info_partial(manager: DatasetManager):
    manager.update_dataset_info("ds1", name="only name")
    reloaded = manager._get_dataset_info("ds1")
    assert reloaded.name == "only name"
    assert reloaded.description == "test dataset"


def test_update_version_info_persists(manager: DatasetManager):
    updated = manager.update_version_info("ds1", "v2", name="v2-renamed")
    assert updated.name == "v2-renamed"

    reloaded = manager._get_version_info("ds1", "v2")
    assert reloaded.name == "v2-renamed"


def test_update_version_info_not_found(manager: DatasetManager):
    with pytest.raises(VersionNotFoundError):
        manager.update_version_info("ds1", "missing", name="x")


# ================ список файлов версии ======================

def test_get_version_files(manager: DatasetManager):
    result = manager.get_version_files("ds1", "v1")
    assert result.total == 3
    assert "train/cat/img1.jpg" in result.files


def test_get_version_files_split_filter(manager: DatasetManager):
    result = manager.get_version_files("ds1", "v1", split=SplitType.TRAIN)
    assert result.total == 3
    assert result.split == "train"
    assert all(f.startswith("train/") for f in result.files)


def test_get_version_files_pagination(manager: DatasetManager):
    result = manager.get_version_files("ds1", "v1", limit=1, offset=1)
    assert result.total == 3
    assert len(result.files) == 1


# ================ integrity ======================

def test_get_version_integrity_from_hashes_file(manager: DatasetManager):
    hashes_path = manager._fsm.worker_path / "ds1" / "v1.hashes.json"
    hashes_path.write_text(json.dumps({
        "train/cat/img1.jpg": "h1",
        "test/cat/imgX.jpg": "h1",
    }), encoding="utf-8")

    report = manager.get_version_integrity("ds1", "v1")
    assert report.summary.leakage.train_test == 1


def test_get_version_integrity_unavailable(manager: DatasetManager):
    with pytest.raises(IntegrityReportNotAvailableError):
        manager.get_version_integrity("ds1", "v1")


def test_drop_version_removes_hashes_file(manager: DatasetManager):
    hashes_path = manager._fsm.worker_path / "ds1" / "v2.hashes.json"
    hashes_path.write_text("{}", encoding="utf-8")

    manager.drop_version("ds1", "v2")

    assert not hashes_path.exists()
    assert not (manager._fsm.worker_path / "ds1" / "v2").exists()


# ================ создание датасета/версии (на cwd-корне) ======================

CLASSES = ("cat", "dog")

def _build_upload_data(root: Path) -> None:
    shade = 0
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            shade += 25
            path = root / split / cls / f"img_{shade}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (shade, 0, 0)).save(path)


@pytest.fixture
def cwd_manager(tmp_path: Path, monkeypatch) -> DatasetManager:
    """DatasetManager на дефолтном корне cwd/datasets с данными в temp/new_data"""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "datasets"
    (root / "temp").mkdir(parents=True)
    _build_upload_data(root / "temp" / "new_data")
    return DatasetManager()


def _new_version(id_data: str = "new_data", name: str = "v_new") -> NewVersion:
    return NewVersion(id_data=id_data, name=name, description="test", sources=[])


def test_add_new_dataset_creates_files(cwd_manager: DatasetManager):
    dsn = NewDataset(
        name="fresh", description="d", type="image", task="classification",
        version=_new_version()
    )
    assert cwd_manager.add_new_dataset(dsn) is True

    root = cwd_manager._fsm.worker_path
    dataset_ids = cwd_manager.get_datasets_id()
    assert len(dataset_ids) == 1
    dataset_path = root / dataset_ids[0]

    dsm = cwd_manager._get_dataset_info(dataset_ids[0])
    version_id = dsm.default_version_id
    assert (dataset_path / version_id / "train" / "cat").is_dir()
    assert (dataset_path / f"{version_id}.hashes.json").is_file()
    assert not (root / "temp" / "new_data").exists()


def test_add_new_version_rollback_on_metadata_failure(cwd_manager: DatasetManager, monkeypatch):
    # сначала создаём датасет штатно
    dsn = NewDataset(
        name="fresh", description="d", type="image", task="classification",
        version=_new_version(name="v1")
    )
    cwd_manager.add_new_dataset(dsn)
    dataset_id = cwd_manager.get_datasets_id()[0]
    versions_before = {v.id for v in cwd_manager._get_dataset_info(dataset_id).versions}

    # вторая загрузка для новой версии
    root = cwd_manager._fsm.worker_path
    _build_upload_data(root / "temp" / "second_data")

    def _fail(dsm):
        raise MetadataSaveError(dsm.id, "disk error")
    monkeypatch.setattr(cwd_manager, "change_dataset_info", _fail)

    with pytest.raises(MetadataSaveError):
        cwd_manager.add_new_version(dataset_id, _new_version(id_data="second_data", name="v2"))

    # данные возвращены в temp, новых папок версий и хешей не осталось
    assert (root / "temp" / "second_data" / "train").is_dir()
    dataset_path = root / dataset_id
    version_dirs = {p.name for p in dataset_path.iterdir() if p.is_dir()}
    assert version_dirs == versions_before
    hashes_files = {p.name for p in dataset_path.glob("*.hashes.json")}
    assert hashes_files == {f"{v}.hashes.json" for v in versions_before}
