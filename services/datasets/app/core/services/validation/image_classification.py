from uuid import uuid4
from PIL import Image
from typing import Dict, Tuple, List, Optional

from app.core.filesystem.fsm import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata, Version, SplitType, ClassDistribution, Split
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.exception.dataset import DatasetValidationError
from app.core.exception.version import VersionValidationError
from app.logs import get_logger

logger = get_logger(__name__)

def dataset_validation_and_create_metadata(
        fsm: FileSystemManager,
        dsn: NewDataset,
    ) -> DatasetMetadata:
        """
        Валидация датасета и вывод метаданных

        * fsm - должен находиться в папке версии
        к примеру в `apple/` и видеть папки `train`, `val` и `test`
        """
        version, new_classes = _version_validation_and_create_metadata(
            dsn.version,
            fsm
        )

        classes_info = {
            "id": str(uuid4()),
            "classes_names": new_classes,
            "classes_count": len(new_classes),
            "classes_to_idx": {class_: indx for indx, class_ in enumerate(sorted(new_classes))},
        }

        return DatasetMetadata(
            **dsn.model_dump(exclude={"version"}),
            **classes_info,
            default_version_id=version.id,
            versions=[version]
        )

# =========================================================================
# __WARNING__ ОБЯЗАТЕЛЬНО РАЗГРУЗИТЬ ЭТУ ФУНКЦИЮ НА НЕСКОЛЬКО МЕЛКИХ
# =========================================================================

def version_validation_and_create_metadata(
    fsm: FileSystemManager,
    nv: NewVersion,
    dsm: DatasetMetadata
) -> Version:
    """
    Валидация версии и вывод метаданных
    * fsm - должен находиться в папке версии
        к примеру в `apple/` и видеть папки `train`, `val` и `test`
    """
    version, _ = _version_validation_and_create_metadata(nv, fsm, dsm)
    return version

def _version_validation_and_create_metadata(
    nv: NewVersion,
    fsm: FileSystemManager,
    dsm: Optional[DatasetMetadata] = None
) -> Tuple[Version, List[str]]:
    """
    Валидация данных
    
    * fsm - должен находиться в папке версии
        к примеру в `apple/` и видеть папки `train`, `val` и `test`
    """
    logger.debug("⬜Валидация данных...")
    try: 
        dir_selections = {'train', 'test', 'val'}
        dir_actual_selections = set(fsm.get_all_dirs())

        # Проверка наличия требуемых выборок
        if dir_actual_selections != dir_selections:
            raise VersionValidationError(
                f"Полученные папки {dir_actual_selections}. Требуемые папки {dir_selections}",
                nv.id_data
            )
        
        fsm.in_dir("train")
        class_names = fsm.get_all_dirs()
        logger.info(
            f"Найденые классы в тренировочной выборке: {class_names}. "
            "В дальнейшем будут использоваться как эталон для проверки"
        )
        fsm.out_dir()

        # Объекты для формирования метаданных
        splits: Dict[SplitType, Split] = {}
        image_format_stats: Dict[str, int] = {}
        # Словарь для хранения общего количества файлов в каждом сплите (нужно для расчета процентов)
        split_total_files = {}

        logger.info('⬜ Проверка структуры:')
        for dir_selection in dir_selections:
            fsm.in_dir(dir_selection)
            dir_classes = fsm.get_all_dirs()

            if set(dir_classes) != set(class_names):
                raise VersionValidationError(
                    "Название папок должно совпадать с названием классов."+
                    f"Получено: {dir_classes}. Требуются: {class_names}",
                    nv.id_data
                )

            # Сбор данных по каждой выборке train/test/val
            count_files_in_selection = 0
            split_type = SplitType(dir_selection)

            # Сначала проходим по классам для подсчета общего количества файлов в сплите
            for dir_class in dir_classes:
                fsm.in_dir(dir_class)
                count_files_in_class = len(fsm.get_all_files())
                count_files_in_selection += count_files_in_class
                fsm.out_dir()
            
            split_total_files[dir_selection] = count_files_in_selection
            
            # Список для сбора объектов в этом сплите
            class_distributions = []

            # Проход для сбора детальной статистики
            for dir_class in dir_classes:
                fsm.in_dir(dir_class)

                # Проверка лишних папок
                if fsm.get_all_dirs():
                    raise VersionValidationError(
                        f"В папке {dir_selection}/{dir_class} найдены лишние папки.",
                        nv.id_data
                    )

                # Проверка существования каких-либо файлов
                files = fsm.get_all_files()
                count_files_in_class = len(files)
                if count_files_in_class == 0:
                    raise DatasetValidationError(
                        f"В {dir_selection}/{dir_class} отсутствуют изображения.",
                        nv.id_data
                    )
                
                # Проверка, что все файлы изображения
                image_files = fsm.all_file_is_image()
                
                # Сбор статистики изображений
                size_counter = {}
                for img_path in image_files:
                    try:
                        with Image.open(img_path) as img:
                            size_str = f"{img.width}x{img.height}"
                            size_counter[size_str] = size_counter.get(size_str, 0) + 1
                    except Exception as e:
                        logger.warning(f"Не удалось получить размер изображения {img_path.name}: {e}")
                
                # Расчет процента от общего количества в выборке
                total_in_split = split_total_files[dir_selection]
                percentage = (count_files_in_class / total_in_split * 100) if total_in_split > 0 else 0
                
                # Создаем объект ClassDistribution
                class_dist = ClassDistribution(
                    class_name=dir_class,
                    class_id=class_names.index(dir_class),
                    count=count_files_in_class,
                    percentage=round(percentage, 2),
                    image_size_count=size_counter
                )
                class_distributions.append(class_dist)
                
                # Сбор статистики форматов изображений
                for img_path in image_files:
                    ext = img_path.suffix.lower().lstrip('.')
                    image_format_stats[ext] = image_format_stats.get(ext, 0) + 1

                logger.debug(f'🟩 {dir_selection}/{dir_class} - {count_files_in_class} изображений')
                fsm.out_dir()

            # Добавляем выборку в словарь splits
            splits[split_type] = Split(class_distribution=class_distributions)
            logger.info(f'🟩 {dir_selection}: {count_files_in_selection} изображений')
            fsm.out_dir()

        size_bytes = fsm.get_dir_size()
        logger.info(f'🟩 Все папки проверены!')
    
    finally:
        fsm.reset()

    version = Version(
        id=str(uuid4()),
        name=nv.name,
        description=nv.description,
        sources=nv.sources,
        size_bytes=size_bytes,
        num_samples=sum(split_total_files.values()),
        image_format_stats=image_format_stats,
        splits=splits
    )

    return version, class_names