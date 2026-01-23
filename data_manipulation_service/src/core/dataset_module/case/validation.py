from pathlib import Path

from logging_ import get_logger
from ..filesystem import FileSystemManager
from core.exception.dataset import DatasetValidationException

logger = get_logger(__name__)


class DatasetImageValidator:
    def __init__(
            self,
            path: Path | None = None
    ):
        _root_path = path if path else (Path.cwd() / "datasets")
        self._fsm = FileSystemManager(_root_path)

    def validate_classification(self):
        """
        Validate image classification dataset structure.

        Expected structure:
            root/
              train/
              valid/
              test/
        """
        logger.debug("Start dataset classification validation")
        self._fsm.reset()

        expected_samples = {"train", "valid", "test"}
        samples = set(self._fsm.get_all_dir())

        if samples != expected_samples:
            raise DatasetValidationException(
                f"Invalid dataset splits: {samples}, expected: {expected_samples}"
            )

        reference_classes: set[str] | None = None

        for sample_name in samples:
            self._fsm.in_dir(sample_name)

            classes = set(self._fsm.get_all_dir())
            if not classes:
                raise DatasetValidationException(
                    f"No classes found in split '{sample_name}'"
                )

            if reference_classes is None:
                reference_classes = classes
            elif classes != reference_classes:
                raise DatasetValidationException(
                    f"Class mismatch in '{sample_name}': "
                    f"{classes} != {reference_classes}"
                )

            for class_name in classes:
                logger.debug(f"Validating '{sample_name}/{class_name}'")
                self._fsm.in_dir(class_name)

                if not self._fsm.all_file_is_image():
                    raise DatasetValidationException(
                        f"Non-image files found in '{sample_name}/{class_name}'"
                    )

                self._fsm.out_dir()

            self._fsm.out_dir()

        logger.info("🟢 Dataset passed classification validation")
