import hashlib
import pytest

from app.core.exception.version import VersionValidationError
from app.core.filesystem import FileSystemManager


# ================ all_file_is_image: служебные файлы ======================

def test_all_file_is_image_skips_junk(fsm: FileSystemManager):
    root = fsm._root
    (root / "img.jpg").write_text("img")
    (root / ".DS_Store").write_text("junk")
    (root / ".hidden").write_text("junk")
    (root / "Thumbs.db").write_text("junk")
    (root / "desktop.ini").write_text("junk")

    files = fsm.all_file_is_image()

    assert {p.name for p in files} == {"img.jpg"}


def test_all_file_is_image_still_raises_on_real_non_image(fsm: FileSystemManager):
    (fsm._root / "img.jpg").write_text("img")
    (fsm._root / "note.txt").write_text("text")

    with pytest.raises(VersionValidationError):
        fsm.all_file_is_image()


# ================ hash_all_files ======================

def test_hash_all_files_returns_sha256_for_images_only(fsm: FileSystemManager):
    root = fsm._root
    (root / "a").mkdir()
    (root / "a" / "x.jpg").write_bytes(b"abc")
    (root / "b.png").write_bytes(b"abc")
    (root / ".DS_Store").write_bytes(b"junk")
    (root / "c.txt").write_bytes(b"text")

    hashes = fsm.hash_all_files()

    assert set(hashes) == {"a/x.jpg", "b.png"}
    expected = hashlib.sha256(b"abc").hexdigest()
    assert hashes["a/x.jpg"] == expected
    assert hashes["b.png"] == expected


def test_hash_all_files_differs_for_different_content(fsm: FileSystemManager):
    (fsm._root / "a.jpg").write_bytes(b"one")
    (fsm._root / "b.jpg").write_bytes(b"two")

    hashes = fsm.hash_all_files()

    assert hashes["a.jpg"] != hashes["b.jpg"]
