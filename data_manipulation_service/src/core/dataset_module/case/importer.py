import shutil
from .validation import DatasetValidator
from core.dataset_module.filesystem import ArchiveManager, FileSystemManager

class DatasetImporter:
    def __init__(
        self,
        datasets_fsm: FileSystemManager,
        archive_manager: ArchiveManager,
    ):
        self._datasets_fsm = datasets_fsm
        self._archive_manager = archive_manager
        self._validator_cls = DatasetValidator

    def import_dataset(
            self, 
            dataset_name: str, 
            archive_name: str
        ):
        # check
        self._datasets_fsm.reset()
        if dataset_name in self._datasets_fsm.get_all_dir():
            raise FileExistsError(f"Dataset '{dataset_name}' already exists")

        # extract
        temp_path = self._archive_manager.extract(archive_name)

        try:
            # validate
            validator = self._validator_cls(temp_path)
            validator.new_dataset()

            # move
            target = self._datasets_fsm._root / dataset_name / "v_0"
            target.parent.mkdir(exist_ok=False)
            shutil.move(str(temp_path), str(target))

        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise
