import hashlib
from pathlib import Path

def has_duplicate_files(
        paths: list[Path], 
        chunk_size: int = 8192
    ) -> bool:
    """
    Check if there are duplicate files by content.
    """
    hashes: set[str] = set()

    for path in paths:
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        file_hash = _hash_file(path, chunk_size)

        if file_hash in hashes:
            return True

        hashes.add(file_hash)

    return False


def _hash_file(
        path: Path, 
        chunk_size: int
    ) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()