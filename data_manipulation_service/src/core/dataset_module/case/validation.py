from pathlib import Path
from shared.logging import get_logger
from ..filesystem import FileSystemManager

logger = get_logger(__name__)

class DatasetValidator:
    def __init__(
            self,
            path: Path | None = None
        ):
        _root_path = path if path else (Path.cwd() / "datasets")
        self._fsm = FileSystemManager(_root_path)

    def new_dataset(self) -> bool:
        """
        Validate dataset structure and class contents.
        """
        logger.debug(f"Start validatetion")
        self._fsm.reset()

        classes = self._fsm.get_all_dir()
        
        for class_ in classes:
            logger.debug(f"Start validatetion class: {class_}")
            self._fsm.in_dir(class_)
            
            if not self._fsm.all_file_is_image():
                return False
            
            self._fsm.out_dir()
        
        logger.info(f"🟢 Dataset passed validation")
        return True