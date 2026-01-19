import hashlib
from logging_ import get_logger
from core.dataset_module.filesystem import FileSystemManager
from pathlib import Path

logger = get_logger(__name__)


class ProcessingFileDuplicate:
    def __init__(
            self,
            file_system_manager: FileSystemManager,
            chunk_size: int = 8192
    ):
        self.chunk_size = chunk_size
        self._fsm = file_system_manager

    def has_duplicate_files(
            self,
    ) -> bool:
        """
        Check if there are duplicate files by content.
        """
        logger.debug('[start] Check duplicate.')
        paths = self._fsm.get_all_files_path()
        hashes: set[str] = set()

        for path in paths:
            if not path.is_file():
                raise ValueError(f"Not a file: {path}")

            file_hash = self._hash_file(path)

            if file_hash in hashes:
                logger.debug(f'[finish] Check duplicate. (Res:{True})')
                return True

            hashes.add(file_hash)

        logger.debug(f'[finish] Check duplicate. (Res:{False})')
        return False

    def find_duplicate_files(
            self,
    ) -> dict[str, list[Path]]:
        """
        Find duplicate files by content.

        Return:
            list of lists of duplicate file paths
        """
        logger.debug('[start] Find duplicate files')
        seen: dict[str, list[Path]] = {}
        paths = self._fsm.get_all_files_path()

        for path in paths:
            h = self._hash_file(path)
            seen.setdefault(h, []).append(path)

        logger.debug('[finish] Find duplicate files')
        return [files for files in seen.values() if len(files) > 1]

    def remove_duplicates(
            self
    ) -> list[Path]:
        """
        Remove duplicate files, keeping the first occurrence by default.
        """
        logger.debug('[start] Remove duplicates')
        duplicates = self.find_duplicate_files()

        for file_list in duplicates:
            file_list_sorted = sorted(file_list)

            # delete the file
            files_to_remove = file_list_sorted[1:]
            for file_path in files_to_remove:
                logger.debug(f'Delete: {file_path}')
                file_path.unlink()

        logger.debug('[finish] Remove duplicates')

    def _hash_file(
            self,
            path: Path,
    ) -> str:
        """
        Compute SHA256 hash of a file, reading in chunks.
        """
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
