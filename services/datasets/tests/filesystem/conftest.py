import pytest
import zipfile
from io import BytesIO
from pathlib import Path
from app.core.filesystem import FileSystemManager, ArchiveManager

@pytest.fixture
def fsm(tmp_path: Path) -> FileSystemManager:
    """Фикстура с чистым FileSystemManager, привязанным к временной папке"""
    manager = FileSystemManager(root=tmp_path)
    return manager


@pytest.fixture
def populated_fs(fsm: FileSystemManager):
    """Подготовленная структура для многих тестов"""
    root = fsm._root

    (root / "photos").mkdir()
    (root / "docs").mkdir()
    (root / "empty").mkdir()

    (root / "photo1.jpg").write_text("img")
    (root / "photo2.png").write_text("img")
    (root / "doc.pdf").write_text("pdf")
    (root / "text.txt").write_text("text")

    (root / "photos" / "cat.jpg").write_text("cat")
    (root / "photos" / "dog.PNG").write_text("dog")
    (root / "photos" / "note.txt").write_text("note")

    return fsm

@pytest.fixture
def temp_dir(tmp_path: Path):
    """Временная папка для каждого теста"""
    return tmp_path / "archive_tests"


@pytest.fixture
def am(temp_dir):
    """Готовый экземпляр ArchiveManager"""
    temp_dir.mkdir(exist_ok=True)
    return ArchiveManager(temp_dir)


@pytest.fixture
def simple_zip_bytes() -> bytes:
    """Простой валидный zip с двумя файлами"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("file1.txt", b"hello world")
        zf.writestr("folder/file2.jpg", b"fake image")
    buffer.seek(0)
    return buffer.getvalue()

@pytest.fixture
def malicious_zip_bytes() -> bytes:
    """Zip с path traversal (../)"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../../etc/passwd", b"malicious content")
    buffer.seek(0)
    return buffer.getvalue()
