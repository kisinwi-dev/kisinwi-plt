from dataclasses import dataclass
from typing import Optional 

@dataclass
class FileSystemManagerStatus:
    dataset: Optional[str]
    version: Optional[str]
    dataset_class: Optional[str]