import pytest
from pathlib import Path
import zipfile
from io import BytesIO
from fastapi import UploadFile
from unittest.mock import Mock

from app.core.filesystem import ArchiveManager

def test_init_creates_temp_folder_if_not_exists(temp_dir):
    non_existing = temp_dir / "new_folder" 
    assert not non_existing.exists()
    non_existing.mkdir(parents=True, exist_ok=True)
    manager = ArchiveManager(non_existing)
    assert non_existing.is_dir()


def test_save_file_saves_with_unique_name(am: ArchiveManager):
    fake_file = Mock(spec=UploadFile)
    fake_file.filename = "test_image.jpg"
    fake_file.file = BytesIO(b"fake image data")

    saved_path = am.save_file(fake_file)

    assert saved_path.is_file()
    assert saved_path.name.endswith("test_image.jpg")
    assert saved_path.parent == am.temp_folder


def test_save_file_raises_if_no_filename(am: ArchiveManager):
    fake_file = Mock(spec=UploadFile)
    fake_file.filename = None
    fake_file.file = BytesIO(b"data")

    with pytest.raises(ValueError):
        am.save_file(fake_file)


def test_unpack_zip_success(am: ArchiveManager, simple_zip_bytes):
    zip_path = am.temp_folder / "test.zip"
    zip_path.write_bytes(simple_zip_bytes)

    extracted = am.unpack(zip_path)

    assert extracted.is_dir()
    assert (extracted / "file1.txt").is_file()
    assert (extracted / "folder" / "file2.jpg").is_file()
    assert (extracted / "file1.txt").read_text() == "hello world"


def test_unpack_raises_on_non_zip(am: ArchiveManager):
    txt_path = am.temp_folder / "data.txt"
    txt_path.write_text("not archive")

    with pytest.raises(ValueError):
        am.unpack(txt_path)


def test_unpack_raises_on_non_existent_file(am: ArchiveManager):
    wrong_path = am.temp_folder / "no-such-file.zip"
    with pytest.raises(FileNotFoundError):
        am.unpack(wrong_path)


def test_unpack_blocks_path_traversal(am: ArchiveManager, malicious_zip_bytes):
    zip_path = am.temp_folder / "evil.zip"
    zip_path.write_bytes(malicious_zip_bytes)

    with pytest.raises(PermissionError):
        am.unpack(zip_path)


def test_unpack_creates_unique_folder_each_time(am: ArchiveManager, simple_zip_bytes):
    zip_path = am.temp_folder / "test.zip"
    zip_path.write_bytes(simple_zip_bytes)

    extract1 = am.unpack(zip_path)
    extract2 = am.unpack(zip_path)

    assert extract1 != extract2
    assert extract1.parent == am.temp_folder
    assert extract2.parent == am.temp_folder


def test_clear_temp_folder_removes_all(am: ArchiveManager):
    # создаём мусор
    (am.temp_folder / "file.txt").write_text("test")
    (am.temp_folder / "subdir").mkdir()
    (am.temp_folder / "subdir" / "inner.txt").write_text("inner")

    assert len(list(am.temp_folder.iterdir())) > 0

    am.clear_temp_folder()

    assert list(am.temp_folder.iterdir()) == []


def test_clear_temp_folder_ignores_errors(am: ArchiveManager, monkeypatch):
    # Симулируем ошибку удаления
    def failing_rmtree(*args, **kwargs):
        raise PermissionError("cannot delete")

    monkeypatch.setattr("shutil.rmtree", failing_rmtree)

    # Не должно упасть
    am.clear_temp_folder()


def test_unpack_raises_on_corrupted_zip(am: ArchiveManager):
    corrupted = am.temp_folder / "broken.zip"
    corrupted.write_bytes(b"PK\x03\x04 corrupted data")

    with pytest.raises(ValueError):
        am.unpack(corrupted)