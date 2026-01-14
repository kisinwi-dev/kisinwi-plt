import shutil
from pathlib import Path
from core.dataset_module.models import FileSystemManagerStatus

class FileSystemManager:
    def __init__(
            self, 
            root: Path | None = None
        ):
        """
        Initialize FileSystemManager with a root datasets directory.
        """
        self._root = (root or Path.cwd() / "datasets").resolve()
        self.worker_path = self._root

    def in_dir(self, dir_name: str) -> None:
        """
        Change current working directory to a subdirectory.
        """
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
        """
        Move one level up in the directory hierarchy.
        """
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
        """
        Rename a subdirectory in the current directory.
        """
        self._dir_exists(old_name)
        self._rename_obj(old_name, new_name)
    
    def rename_file(
            self,
            old_name: str,
            new_name: str, 
        ):
        """
        Rename a file in the current directory.
        """
        self._file_exists(old_name)
        self._rename_obj(old_name, new_name)
        
    def _rename_obj(
            self, 
            old_name: str,
            new_name: str 
        ): 
        """
        Rename a file system object (file or directory).
        """
        old_path = self.worker_path / old_name
        new_path = self.worker_path / new_name
        if new_path.exists():
            raise FileExistsError(f"Directory '{new_name}' already exists")
        old_path.rename(new_path)

    def drop_dir(
            self,
            dir_name
        ):
        """
        Remove a directory and all its contents recursively.
        """
        self._dir_exists(dir_name)
        path = (self.worker_path / dir_name).resolve()

        if not path.is_relative_to(self._root):
            raise PermissionError("Cannot remove directory outside root")

        shutil.rmtree(path)

    def drop_file(self, file_name: str) -> None:
        """
        Remove a file from the current directory.
        """
        self._file_exists(file_name)
        (self.worker_path / file_name).unlink()

    def _dir_exists(
            self,
            dir_name,
        )-> bool:
        """
        Check if a directory exists in the current directory.
        """
        dirs = self.get_all_dir()
        if dir_name not in dirs:
            raise FileNotFoundError(f"Dir {dir_name} not found")
        return True
    
    def _file_exists(
            self,
            file_name
        )-> bool:
        """
        Check if a file exists in the current directory.
        """
        files = self.get_all_file()
        if file_name not in files:
            raise FileNotFoundError(f"File {file_name} not found")
        return True