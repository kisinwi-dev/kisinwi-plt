from pathlib import Path
from shared.logging import get_logger
from .filesystem import FileSystemManager

logger = get_logger(__name__)

class DatasetValidator:
    def __init__(
            self,
            fsm: FileSystemManager | None = None
        ):
        self._fsm = fsm if fsm else FileSystemManager(Path.cwd() / "datasets").resolve()

    def check_new_dataset(
            self,
            name_dataset: str,
        ) -> bool:
        """
        Validate dataset structure and class contents.
        """
        logger.debug(f"Start validatetion dataset:{name_dataset}")
        self._fsm.reset()
        self._fsm._dir_exists(name_dataset)
        self._fsm.in_dir(name_dataset)

        classes = self._fsm.get_all_dir()
        
        for class_ in classes:
            logger.debug(f"Start validatetion class:{class_}")
            self._fsm.in_dir(class_)
            
            if not self._fsm.all_file_is_image():
                return False
            
            self._fsm.out_dir()
        
        logger.info(f"🟢 Dataset {name_dataset} passed validation")
        return True