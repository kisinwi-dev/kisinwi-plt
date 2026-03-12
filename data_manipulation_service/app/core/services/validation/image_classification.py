from app.core.exception.dataset import DatasetValidationError
from app.core.filesystem.fsm import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata, Version
from app.api.schemas.dataset_new import NewDataset
from app.logs import get_logger

logger = get_logger(__name__)

def validation_and_create_metadata(
        dsn: NewDataset,
        fsm: FileSystemManager
    ) -> DatasetMetadata:
        "Валидация датасета"
        
        fsm.in_dir("temp")

        if dsn.dataset_id not in fsm.get_all_dirs():
            logger.debug(f'🩸Path: {fsm.worker_path}')
            raise FileNotFoundError(f"Не найден dataset с dataset_id = {dsn.dataset_id}")
        
        try:    
            fsm.in_dir(dsn.dataset_id)

            dir_selections = {'train', 'test', 'val'}
            dir_actual_selections = set(fsm.get_all_dirs())

            if dir_actual_selections != dir_selections:
                raise DatasetValidationError(
                    f"Полученные папки {dir_actual_selections}. Требуемые папки {dir_selections}",
                    dsn.dataset_id
                )
            
            num_samples = dict()

            logger.info('| ⬜ Проверка структуры:')
            for dir_selection in dir_selections:
                fsm.in_dir(dir_selection)
                dir_classes = fsm.get_all_dirs()

                if set(dir_classes) != set(dsn.class_names):
                    raise DatasetValidationError(
                        "Название папок должно совпадать с названием классов."+
                        f"Получено: {dir_classes}. Требуются: {dsn.class_names}",
                        dsn.dataset_id
                    )

                count_files_in_selection = 0

                for dir_class in dir_classes:
                    fsm.in_dir(dir_class)

                    if fsm.get_all_dirs():
                        raise DatasetValidationError(
                            f"В папке {dir_selection}/{dir_class} найдены лишние папки.",
                            dsn.dataset_id
                        )

                    if not fsm.all_file_is_image():
                        raise DatasetValidationError(
                            f"В папке {dir_selection}/{dir_class} не все файлы являются изображениями.",
                            dsn.dataset_id
                        )
                    
                    count_files_in_class = len(fsm.get_all_files())
                    if count_files_in_class == 0:
                        raise DatasetValidationError(
                            f"В {dir_selection}/{dir_class} остуствуют изображения.",
                            dsn.dataset_id
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
        
        
        classes_info = {
            "num_classes": len(dsn.class_names),
            "class_to_idx": {class_: indx for indx, class_ in enumerate(sorted(dsn.class_names))},
        }
        
        version = Version(
            **dsn.version.model_dump(),
            size_bytes = size_bytes,
            num_samples = sum(num_samples.values()),
            num_train = num_samples['train'],
            num_val = num_samples['val'],
            num_test = num_samples['test']
        )

        return DatasetMetadata(
            **dsn.model_dump(exclude={"version"}),
            **classes_info,
            default_version_id=version.version_id,
            versions=[version]
        )