from dataclasses import dataclass
from typing import Optional 

@dataclass
class FileSystemManagerStatus:
    dataset: Optional[str]
    version: Optional[str]
    dataset_class: Optional[str]

@dataclass
class ClassInfo:
    name: str 
    description: Optional[str]
    count_files: Optional[int]
    type_files: Optional[str]

@dataclass
class VersionInfo:
    name: str
    description: Optional[str]
    classes: list[ClassInfo]

@dataclass
class DatasetInfo:
    name: str
    description: Optional[str]
    versions: list[VersionInfo]