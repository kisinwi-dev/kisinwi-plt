from typing import Callable, Dict, Tuple, List

from app.api.schemas.dataset import DatasetMetadata, Version
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.filesystem.fsm import FileSystemManager

from .image_classification import (
    dataset_validation_and_create_metadata as dvm,
    version_validation_and_create_metadata as vvm
)

PreprocLoaderDataset = Callable[[NewDataset, FileSystemManager], DatasetMetadata]

PREPROC_LOADERS_DATASET: Dict[Tuple[str, str], PreprocLoaderDataset] = {
    ("image", "classification"): dvm,
}

PreprocLoaderVersion = Callable[[List[str], NewVersion, FileSystemManager], Version]

PREPROC_LOADERS_VERSION: Dict[Tuple[str, str], PreprocLoaderVersion] = {
    ("image", "classification"): vvm,
}

def type_task_supported() -> list[Tuple[str, str]]: 
    return list(PREPROC_LOADERS_DATASET.keys())