from uuid import uuid4
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from app.core.filesystem.fsm import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata, Version, SplitType
from app.api.schemas.splits import ClassDistribution, Split
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.exception.dataset import DatasetValidationError
from app.core.exception.version import VersionValidationError
from app.logs import get_logger

logger = get_logger(__name__)

SELECTION_NAME = {item.value for item in SplitType}

def dataset_validation_and_create_metadata(
        fsm: FileSystemManager,
        dsn: NewDataset,
    ) -> DatasetMetadata:
        """
        Валидация датасета и вывод метаданных

        * fsm - должен находиться в временной папке в новых данных.
        К примеру в `temp/my_new_data/` и видеть папки `train`, `val` и `test`
        """
        version, new_classes = _data_validation_and_create_metadata(
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

def version_validation_and_create_metadata(
    fsm: FileSystemManager,
    nv: NewVersion,
    dsm: DatasetMetadata
) -> Version:
    """
    Валидация версии и вывод метаданных
    * fsm - должен находиться в временной папке в новых данных.
    К примеру в `temp/my_new_data/` и видеть папки `train`, `val` и `test`
    """
    version, _ = _data_validation_and_create_metadata(nv, fsm, dsm)
    return version

def _data_validation_and_create_metadata(
    nv: NewVersion,
    fsm: FileSystemManager,
    dsm: Optional[DatasetMetadata] = None
) -> Tuple[Version, List[str]]:
    """
    Валидация данных.
    
    Работает в 2 итерации. Первый раз проверяет структуру данных, после чего проверяет сами данные.
    
    * fsm - должен находиться в временной папке в новых данных.
    К примеру в `temp/my_new_data/` и видеть папки `train`, `val` и `test`
    """
    logger.debug("⬜ Валидация данных...")
    try:
        # загрузка списка классов для проверки датасета
        class_names = _classes_load(fsm, dsm)

        # проверка структуры
        _check_cache(fsm, class_names)

        # расчёт статистики
        splits, image_format_stats = _calculate_statistics(fsm, class_names)

        # расчёт размера в байтах
        size_bytes = fsm.get_dir_size()
    
    finally:
        fsm.reset()

    version = Version(
        id=str(uuid4()),
        name=nv.name,
        description=nv.description,
        sources=nv.sources,
        size_bytes=size_bytes,
        image_format_stats=image_format_stats,
        splits=splits
    )

    return version, class_names

def _calculate_statistics(
    fsm: FileSystemManager,
    class_names: List[str]
) -> Tuple[Dict[SplitType, Split], Dict[str, int]]:
    """
    Расчёт статистики данных.

    Args:
        fsm: файловый менеджер
        class_names: список имён классов

    Returns:
        * словарь выборок, где ключ имя выборки, а значение статистика по выборке
        * словарь разрешений, где ключ это разрешение, а значение это количество
    """
    # Объекты для формирования метаданных
    splits: Dict[SplitType, Split] = {}
    image_format_stats: Dict[str, int] = {}

    # Проверка самих файлов и статистики
    for selection in SELECTION_NAME:
        fsm.in_dir(selection)
        dir_classes = fsm.get_all_dirs()

        # тип сплита train/val/test
        split_type = SplitType(selection)
        # Список для сбора объектов метаданных по распределениям классов
        class_distributions: List[ClassDistribution] = []

        # Проход для сбора детальной статистики
        for dir_class in dir_classes:
            fsm.in_dir(dir_class)

            # Получение статистики по всем изображениям в директории fsm
            image_statistics = _class_statistic(fsm)
            # пополнение словаря форматов
            image_format_stats = image_format_stats | image_statistics.format_stats

            # Создаем объект ClassDistribution
            class_dist = ClassDistribution(
                class_name=dir_class,
                class_id=class_names.index(dir_class),
                count=image_statistics.image_count,
                image_size_count=image_statistics.size_counter
            )
            class_distributions.append(class_dist)

            logger.debug(f'🟩 `{selection}/{dir_class}` - `{image_statistics.image_count}` изображений')
            fsm.out_dir()
        
        # расчёт процентного соотношения классов к всей выборке
        _calculate_percentage_in_all(class_distributions)

        # Добавляем выборку в словарь splits
        splits[split_type] = Split(class_distribution=class_distributions)
        logger.info(f'🟩 Файлы в `{selection}` успешно проверены')
        fsm.out_dir()

    logger.info(f'✅ Все папки проверены!')
    return splits, image_format_stats

def _check_cache(
    fsm: FileSystemManager,
    class_names: List[str]
):
    """
    Проверка структуры полученных данных
    
    Args:
        fsm: обьект файлового менеджера в директории проверки
        class_names: список имён классов
    """
    logger.info('⬜ Проверка структуры:')
    id_data = fsm.worker_path.name

    # Проверка наличия требуемых выборок
    dir_actual_selections = set(fsm.get_all_dirs())
    if dir_actual_selections != SELECTION_NAME:
        raise VersionValidationError(
            f"Полученные папки `{dir_actual_selections}`. Требуемые папки `{SELECTION_NAME}`",
            id_data
        )
    logger.debug(f'🟩 Существует выборка {SELECTION_NAME}')
    
    # проход по выборкам
    for selection in SELECTION_NAME:
        fsm.in_dir(selection)
        classes_name = fsm.get_all_dirs()

        if set(classes_name) != set(class_names):
            raise VersionValidationError(
                "Название папок должно совпадать с названием классов."+
                f"Получено: `{classes_name}`. Требуются: `{class_names}`",
                id_data
            )
        logger.debug(f'-🟩 Классы в выборке `{selection}` соответсвуют требуемым')

        # проход по классам
        for class_name in classes_name:
            fsm.in_dir(class_name)

            # Проверка мусорных файлов и наличия файлов
            if fsm.get_all_dirs():
                raise VersionValidationError(
                    f"В папке `{selection}/{class_name}` найдены лишние папки.",
                    id_data
                )
            
            # Проверка существования каких-либо файлов
            files = fsm.all_file_is_image()
            count_files_in_class = len(files)
            if count_files_in_class == 0:
                raise DatasetValidationError(
                    f"В `{selection}/{class_name}` отсутствуют изображения.",
                    id_data
                )
            logger.debug(f'--🟩 Стуруктура в `{selection}/{class_name}`')
            fsm.out_dir()
        fsm.out_dir()
    logger.info("✅ Структура проверена")

def _calculate_percentage_in_all(
    class_distributions: List[ClassDistribution]
):
    """Расчёт процентного отношения между классами"""
    total_files = sum(cd.count for cd in class_distributions)
    for cd in class_distributions:
        cd.calculate_percentage(total_files)

@dataclass
class ImageStatistics():
    image_count: int = 0
    size_counter: Dict[str, int] = field(default_factory=dict)
    format_stats: Dict[str, int] = field(default_factory=dict)

def _class_statistic(
    fsm: FileSystemManager,
) -> ImageStatistics:
    """
    Сбор статистики по всем изображениям
    
    Args:
        fsm: файловый менеджер
    """
    # Получение path всех изображений
    image_files = fsm.all_file_is_image()
    
    image_statistics = ImageStatistics(
        image_count=len(image_files)
    )

    for img_path in image_files:
        try:
            # Статистика разрешений
            with Image.open(img_path) as img:
                size_str = f"{img.width}x{img.height}"
                image_statistics.size_counter[size_str] = image_statistics.size_counter.get(size_str, 0) + 1

            # Статистика форматов
            ext = img_path.suffix.lower().lstrip('.')
            image_statistics.format_stats[ext] = image_statistics.format_stats.get(ext, 0) + 1
        except Exception as e:
            logger.warning(f"Не удалось получить размер изображения {img_path.name}: {e}")
    return image_statistics

def _classes_load(
    fsm: FileSystemManager,
    dsm: Optional[DatasetMetadata] = None
) -> List[str]:
    """Загрузка списка классов"""
    if dsm is None:
        fsm.in_dir("train")
        class_names = fsm.get_all_dirs()
        logger.info(
            f"Найденые классы в тренировочной выборке: {class_names}. "
            "В дальнейшем будут использоваться как эталон для проверки"
        )
        fsm.out_dir()
        return class_names
    else:
       return dsm.classes_names 
