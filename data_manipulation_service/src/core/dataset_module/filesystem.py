from pathlib import Path
from core.dataset_module.models import FileSystemManagerStatus

class FileSystemManager:
    def __init__(self, root: Path | None = None):
        self._root = (root or Path.cwd() / "datasets").resolve()
        self.worker_path = self._root

    def in_dir(self, dir_name: str) -> None:
        new_path = (self.worker_path / dir_name).resolve()
        
        if not new_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_name}")
        
        if not new_path.is_relative_to(self._root):
            raise PermissionError("Cannot leave root directory")
        
        self.worker_path = new_path

    def in_dirs(
            self, 
            dirs_list: list[str]
        ):
        """
        Move sequentially into a list of subdirectories.
        """
        for dir in dirs_list:
            self.in_dir(dir)

    def out_dir(self) -> None:
        
        if self.worker_path == self._root:
            raise PermissionError("Cannot go above root directory")
        
        self.worker_path = self.worker_path.parent

    def status(self) -> FileSystemManagerStatus:
        """
        Return information about the current position in the datasets directory.

        Structure is inferred purely from the directory hierarchy:
        datasets/{dataset_name}/v_{version}/{class_name}/
        """
        relative = self.worker_path.relative_to(self._root)
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
        self.worker_path = self._root

    def rename_dir(
            self, 
            old_name: str, 
            new_name: str
        ):
        dir_list = self.get_all_dir()
        if old_name not in dir_list:
            raise FileNotFoundError(f"Dir {old_name} not found")
        self._rename_obj(old_name, new_name)
    
    def rename_file(
            self,
            old_name: str,
            new_name: str, 
        ):
        file_list = self.get_all_file()
        if old_name not in file_list:
            raise FileNotFoundError(f"Dir {old_name} not found")
        self._rename_obj(old_name, new_name)
        
    def _rename_obj(
            self, 
            old_name: str,
            new_name: str 
        ): 
        old_path = Path(self.worker_path / old_name)
        new_path = Path(self.worker_path / new_name)
        old_path.rename(new_path)