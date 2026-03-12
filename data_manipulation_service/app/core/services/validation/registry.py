from typing import Callable, Dict, Tuple

from app.api.schemas.dataset import DatasetMetadata
from app.api.schemas.dataset_new import NewDataset
from app.core.filesystem.fsm import FileSystemManager

from .image_classification import validation_and_create_metadata

PreprocLoader = Callable[[NewDataset, FileSystemManager | None], DatasetMetadata]

PREPROC_LOADERS: Dict[Tuple[str, str], PreprocLoader] = {
    ("image", "classification"): validation_and_create_metadata,
}

def type_task_supported() -> list[Tuple[str, str]]: 
    return list(PREPROC_LOADERS.keys())