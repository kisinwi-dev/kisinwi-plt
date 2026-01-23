import shutil
from pathlib import Path
from .validation import DatasetImageValidator
from core.dataset_module.filesystem import FileSystemManager, TempManager


class Importer:
    def __init__(
            self,
            file_system_manager: FileSystemManager,
            temp_manager: TempManager
    ):
        self._datasets_fsm = file_system_manager
        self._temp_manager = temp_manager

    def import_dataset(
            self,
            dataset_name: str,
            dataset_type: str,
            dataset_task: str,
            path_archive: Path
    ):
        self._datasets_fsm.reset()

        if dataset_name in self._datasets_fsm.get_all_dir():
            raise FileExistsError(f"Dataset '{dataset_name}' already exists")

        temp_path = self._temp_manager.extract(path_archive)

        try:
            # valid
            self._validate_dataset(dataset_type, dataset_task, temp_path)

            # move
            target = self._datasets_fsm._root / dataset_name / "v_0"
            target.parent.mkdir(exist_ok=False)
            shutil.move(str(temp_path), str(target))

        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise

    def _validate_dataset(
            self,
            dataset_type: str,
            dataset_task: str,
            temp_path: Path
    ):
        if dataset_type == "image":

            if dataset_task == "classification":
                validator = DatasetImageValidator(temp_path)
                validator.validate_classification()
                return            
            else:
                raise UnsupportedDatasetTaskError(
                    f"Unsupported image task: '{dataset_task}'"
                )
        else:
            raise UnsupportedDatasetTypeError(
                f"Unsupported dataset type: '{dataset_type}'"
            )


class DatasetImportError(Exception):
    pass


class UnsupportedDatasetTypeError(DatasetImportError):
    pass


class UnsupportedDatasetTaskError(DatasetImportError):
    pass
