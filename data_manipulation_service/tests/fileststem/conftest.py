import pytest
import zipfile
from io import BytesIO
from pathlib import Path
from app.core.filesystem import FileSystemManager, ArchiveManager

@pytest.fixture
def temp_dir(tmp_path: Path):
    """Временная папка для каждого теста"""
    folder = tmp_path / "test_folder"
    folder.mkdir(exist_ok=True)
    return folder

@pytest.fixture
def fsm(temp_dir: Path) -> FileSystemManager:
    """Фикстура с чистым FileSystemManager, привязанным к временной папке"""
    manager = FileSystemManager(root=temp_dir)
    return manager


@pytest.fixture
def populated_fs(fsm: FileSystemManager):
    """Подготовленная структура для многих тестов"""
    root = fsm._root

    # файлы лежащие в корне
    (root / "photo1.jpg").write_text("img")
    (root / "photo2.png").write_text("img")
    (root / "text.txt").write_text("text")

    # пустая папка
    (root / "empty").mkdir()

    # папка с 1 документом
    (root / "docs").mkdir()
    (root / "doc.pdf").write_text("pdf")

    # папка с изображениями и .txt
    (root / "photos").mkdir()
    (root / "photos" / "cat.jpg").write_text("cat")
    (root / "photos" / "dog.PNG").write_text("dog")
    (root / "photos" / "note.txt").write_text("note")

    # папка толко с изображениями
    (root / "only_photos").mkdir()
    (root / "only_photos" / "cat.jpg").write_text("cat")
    (root / "only_photos" / "dog.jpg").write_text("dog")

    return fsm


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
