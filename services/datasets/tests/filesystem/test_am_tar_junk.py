import os
import tarfile
import time
import zipfile
from io import BytesIO

import pytest

from app.core.filesystem import ArchiveManager


def _tar_bytes(names: list[str], mode: str) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as tf:
        for name in names:
            data = name.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
    buffer.seek(0)
    return buffer.getvalue()


# ================ распаковка tar ======================

def test_unpack_tar_gz(am: ArchiveManager):
    archive = am.temp_folder / "data.tar.gz"
    archive.write_bytes(_tar_bytes(["train/cat/img1.jpg", "val/cat/img2.jpg"], "w:gz"))

    extracted = am.unpack(archive, "tester")

    assert (extracted / "train" / "cat" / "img1.jpg").is_file()
    assert (extracted / "val" / "cat" / "img2.jpg").is_file()
    assert not archive.exists()


def test_unpack_plain_tar(am: ArchiveManager):
    archive = am.temp_folder / "data.tar"
    archive.write_bytes(_tar_bytes(["train/cat/img1.jpg", "test/cat/img2.jpg"], "w"))

    extracted = am.unpack(archive, "tester")

    assert (extracted / "train" / "cat" / "img1.jpg").is_file()
    assert (extracted / "test" / "cat" / "img2.jpg").is_file()


def test_unpack_unsupported_format(am: ArchiveManager):
    archive = am.temp_folder / "data.rar"
    archive.write_bytes(b"not really an archive")

    with pytest.raises(ValueError):
        am.unpack(archive, "tester")


def test_unpack_corrupted_tar_gz(am: ArchiveManager):
    archive = am.temp_folder / "data.tar.gz"
    archive.write_bytes(b"definitely not a tar.gz")

    with pytest.raises(ValueError):
        am.unpack(archive, "tester")


# ================ __MACOSX ======================

def test_unpack_removes_macosx_and_flattens(am: ArchiveManager):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("my_dataset/train/cat/img1.jpg", b"fake image")
        zf.writestr("__MACOSX/my_dataset/._img1.jpg", b"resource fork")
    archive = am.temp_folder / "mac.zip"
    archive.write_bytes(buffer.getvalue())

    extracted = am.unpack(archive, "tester")

    assert not (extracted / "__MACOSX").exists()
    # после удаления __MACOSX обёртка my_dataset считается единственным корнем и снимается
    assert (extracted / "train" / "cat" / "img1.jpg").is_file()


# ================ cleanup_stale ======================

def test_cleanup_stale_removes_only_old_items(am: ArchiveManager):
    old_dir = am.temp_folder / "old_upload"
    old_dir.mkdir()
    fresh_dir = am.temp_folder / "fresh_upload"
    fresh_dir.mkdir()

    old_time = time.time() - 48 * 3600
    os.utime(old_dir, (old_time, old_time))

    removed = am.cleanup_stale(ttl_hours=24)

    assert removed == 1
    assert not old_dir.exists()
    assert fresh_dir.exists()
