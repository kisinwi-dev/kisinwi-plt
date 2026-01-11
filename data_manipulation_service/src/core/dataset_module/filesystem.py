import os 
from pathlib import Path

class FileSystemManager:
    def __init__(self):
        self.root = Path(str(Path.cwd()) + "\\datasets")

    def in_dir(self, dir: str):
        self.root = Path(self.root / dir)

    def status(self):
        right_path = str(self.root).split('\\src\\')[-1]
        return right_path

    def get_all_dir(self):
        return [str(path).split('\\')[-1] for path in self.root.iterdir() if path.is_dir()]
    
    def get_all_file(self):
        return [str(path).split('\\')[-1] for path in self.root.iterdir() if path.is_file()]