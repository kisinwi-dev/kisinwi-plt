from pathlib import Path


class BaseStorage:
    def __init__(self, base_path: str = "discussion"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
