from pathlib import Path

class ArchiveManager:
    def __init__(
            self, 
            root: Path | None = None
        ):
        self._root = (root or Path.cwd() / "temp").resolve()
        self.worker_path = self._root

    def extract(
            self,
            archive_name: str
        ) -> Path:
        archive_path = self._root / archive_name
        if not archive_path.exists() or not archive_path.is_file():
            raise FileNotFoundError(f"{archive_name} not found")
        
        suffix = archive_path.suffix.lower()
        if suffix == ".zip":
            return self._extract_zip(archive_path)
        else:
            raise ValueError(f"Unsupported archive type: {suffix}")
        
    def _extract_zip(self, archive_path: Path) -> Path:
        import zipfile
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            self._safe_extract(zip_ref.namelist(), self._root)
            zip_ref.extractall(self._root)
        return self._root
    
    def _safe_extract(self, members: list[str], target: Path):
        for member in members:
            member_path = (target / member).resolve()
            if not member_path.is_relative_to(target):
                raise PermissionError("Unsafe archive content detected")