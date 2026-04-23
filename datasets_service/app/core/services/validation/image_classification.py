from app.core.filesystem.fsm import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata, Version
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.exception.dataset import DatasetValidationError
from app.core.exception.version import VersionValidationError
from app.logs import get_logger

logger = get_logger(__name__)

def dataset_validation_and_create_metadata(
        dsn: NewDataset,
        fsm: FileSystemManager
    ) -> DatasetMetadata:
        """
        Валидация датасета и вывод метаданных по версии
        * fsm - должен находиться в папке где лежит датасет
            к примеру в `temp/` и видеть папку датасета `apple`
        """

        if dsn.dataset_id not in fsm.get_all_dirs():
            logger.debug(f'🩸Path: {fsm.worker_path}')
            raise FileNotFoundError(f"Не найден dataset с dataset_id = {dsn.dataset_id}")
       
        fsm.in_dir(dsn.dataset_id)
        version = version_validation_and_create_metadata(
            dsn.class_names, 
            dsn.version,
            fsm
        )
        
        classes_info = {
            "num_classes": len(dsn.class_names),
            "class_to_idx": {class_: indx for indx, class_ in enumerate(sorted(dsn.class_names))},
        }

        return DatasetMetadata(
            **dsn.model_dump(exclude={"version"}),
            **classes_info,
            default_version_id=version.version_id,
            versions=[version]
        )

def version_validation_and_create_metadata(
    class_names: list[str],
    nv: NewVersion,
    fsm: FileSystemManager
) -> Version:
    """
    Валидация версии и вывод метаданных по версии
    * class_names - ожидаемые классы находящиеся в версии
    * fsm - должен находиться в папке версии
        к примеру в `apple/` и видеть папки `train`, `val` и `test`
    """
    logger.debug('Валидация версии')
    try: 
        dir_selections = {'train', 'test', 'val'}
        dir_actual_selections = set(fsm.get_all_dirs())

        if dir_actual_selections != dir_selections:
            raise VersionValidationError(
                f"Полученные папки {dir_actual_selections}. Требуемые папки {dir_selections}",
            )
        
        num_samples = dict()

        logger.info('| ⬜ Проверка структуры:')
        for dir_selection in dir_selections:
            fsm.in_dir(dir_selection)
            dir_classes = fsm.get_all_dirs()

            if set(dir_classes) != set(class_names):
                raise VersionValidationError(
                    "Название папок должно совпадать с названием классов."+
                    f"Получено: {dir_classes}. Требуются: {class_names}",
                    nv.version_id
                )

            count_files_in_selection = 0

            for dir_class in dir_classes:
                fsm.in_dir(dir_class)

                if fsm.get_all_dirs():
                    raise VersionValidationError(
                        f"В папке {dir_selection}/{dir_class} найдены лишние папки.",
                        nv.version_id
                    )

                fsm.all_file_is_image()    
                
                count_files_in_class = len(fsm.get_all_files())
                if count_files_in_class == 0:
                    raise DatasetValidationError(
                        f"В {dir_selection}/{dir_class} остуствуют изображения.",
                        nv.version_id
                    )
                count_files_in_selection += count_files_in_class

                logger.info(f'| | 🟩 {dir_selection}/{dir_class}')
                fsm.out_dir()

            num_samples[dir_selection] = count_files_in_selection 
            fsm.out_dir()

        size_bytes = fsm.get_dir_size()
        logger.info(f'| 🟩 Все папки проверены!')
    
    finally:
        fsm.reset()

    version = Version(
        **nv.model_dump(),
        size_bytes = size_bytes,
        num_samples = sum(num_samples.values()),
        num_train = num_samples['train'],
        num_val = num_samples['val'],
        num_test = num_samples['test']
    )

    return version
