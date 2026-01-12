from pathlib import Path
from core.dataset_module.models import FileSystemManagerStatus

class FileSystemManager:
    def __init__(self, root: Path | None = None):
        self.root = (root or Path.cwd() / "datasets").resolve()
        self.worker_path = self.root

    def in_dir(self, dir_name: str) -> None:
        new_path = (self.worker_path / dir_name).resolve()
        
        if not new_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_name}")
        
        if not new_path.is_relative_to(self.root):
            raise PermissionError("Cannot leave root directory")
        
        self.worker_path = new_path

    def out_dir(self) -> None:
        
        if self.worker_path == self.root:
            raise PermissionError("Cannot go above root directory")
        
        self.worker_path = self.worker_path.parent

    def status(self) -> FileSystemManagerStatus:
        """
        Return information about the current position in the datasets directory.

        Structure is inferred purely from the directory hierarchy:
        datasets/{dataset_name}/v_{version}/{class_name}/
        """
        relative = self.worker_path.relative_to(self.root)
        parts = relative.parts

        dataset = None
        version = None
        dataset_class = None

        if len(parts) >= 1:
            dataset = parts[0]

        if len(parts) >= 2 and parts[1].startswith("v_"):
            version = parts[1]

        if len(parts) >= 3 and parts[2]:
            dataset_class = parts[2]

        return FileSystemManagerStatus(
            dataset=dataset,
            version=version,
            dataset_class=dataset_class
        )

    def get_all_dir(self) -> list[str]:
        """
        Return names of subdirectories in the current directory.
        """
        return [path.name for path in self.worker_path.iterdir() if path.is_dir()]
    
    def get_all_file(self) -> list[str]:
        """
        Return names of files in the current directory.
        """
        return [path.name for path in self.worker_path.iterdir() if path.is_file()]
    
    def reset(self):
        """
        Reset the current working directory to the root directory.
        """
        self.worker_path = self.root