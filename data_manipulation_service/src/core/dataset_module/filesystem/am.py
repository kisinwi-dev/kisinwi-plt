import zipfile
import uuid
from shared.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

class ArchiveManager:
    def __init__(
            self, 
            root: Path | None = None
        ):
        self._root = (root or Path.cwd() / "temp").resolve()

    def extract(
            self,
            archive_name: str,
        ) -> Path:
        """
        Extract an archive into the root directory.

        Supported formats: ZIP
        """
        logger.debug(f"Start extracting archive: {archive_name}")
        archive_path = self._root / archive_name
        if not archive_path.exists() or not archive_path.is_file():
            raise FileNotFoundError(f"{archive_name} not found")
        
        suffix = archive_path.suffix.lower()
        logger.debug(f"Detected archive type: {suffix}")
        extract_dir_path = self._generate_temp_dir()
        if suffix == ".zip":
            self._extract_zip(archive_path, extract_dir_path)
        else:
            raise ValueError(f"Unsupported archive type: {suffix}")
        logger.info("Archive extracted successfully: %s", archive_name)

        return extract_dir_path

    def _generate_temp_dir(self) -> Path:
        name = f"upload_{uuid.uuid4().hex}"
        path = self._root / name
        path.mkdir(parents=True)
        return path

    def _extract_zip(
            self, 
            archive_path: Path,
            target: Path
        ):
        """
        Safely extract a ZIP archive into the root directory.
        """
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            self._safe_extract(zip_ref.namelist(), target)
            zip_ref.extractall(target)
    
    def _safe_extract(
            self, 
            members: list[str], 
            target: Path
        ):
        """
        Safely extract a ZIP archive into the given target directory.
        """
        for member in members:
            member_path = (target / member).resolve()
            if not member_path.is_relative_to(target):
                logger.error(f"Unsafe archive member detected: {member}")
                raise PermissionError("Unsafe archive content detected")