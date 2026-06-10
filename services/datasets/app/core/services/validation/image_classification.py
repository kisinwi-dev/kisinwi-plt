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
from app.core.services.integrity import compute_integrity_summary
from app.logs import get_logger

logger = get_logger(__name__)

SELECTION_NAME = {item.value for item in SplitType}

def dataset_validation_and_create_metadata(
        fsm: FileSystemManager,
        dsn: NewDataset,
    ) -> Tuple[DatasetMetadata, Dict[str, str]]:
        """
        Валидация датасета и вывод метаданных + карта хешей файлов

        * fsm - должен находиться в временной папке в новых данных.
        К примеру в `temp/my_new_data/` и видеть папки `train`, `val` и `test`
        """
        version, new_classes, hashes = _data_validation_and_create_metadata(
            dsn.version,
            fsm
        )

        classes_info = {
            "id": str(uuid4()),
            "classes_names": new_classes,
            "classes_count": len(new_classes),
            "classes_to_idx": {class_: indx for indx, class_ in enumerate(sorted(new_classes))},
        }

        dsm = DatasetMetadata(
            **dsn.model_dump(exclude={"version"}),
            **classes_info,
            default_version_id=version.id,
            versions=[version]
        )
        return dsm, hashes

def version_validation_and_create_metadata(
    fsm: FileSystemManager,
    nv: NewVersion,
    dsm: DatasetMetadata
) -> Tuple[Version, Dict[str, str]]:
    """
    Валидация версии и вывод метаданных + карта хешей файлов
    * fsm - должен находиться в временной папке в новых данных.
    К примеру в `temp/my_new_data/` и видеть папки `train`, `val` и `test`
    """
    version, _, hashes = _data_validation_and_create_metadata(nv, fsm, dsm)
    return version, hashes

def _data_validation_and_create_metadata(
    nv: NewVersion,
    fsm: FileSystemManager,
    dsm: Optional[DatasetMetadata] = None
) -> Tuple[Version, List[str], Dict[str, str]]:
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
        splits, image_format_stats, color_mode_stats = _calculate_statistics(fsm, class_names)

        # хеши файлов: дубликаты и утечки между сплитами
        hashes = fsm.hash_all_files()

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
        color_mode_stats=color_mode_stats,
        integrity=compute_integrity_summary(hashes),
        splits=splits
    )

    return version, class_names, hashes

def _calculate_statistics(
    fsm: FileSystemManager,
    class_names: List[str]
) -> Tuple[Dict[SplitType, Split], Dict[str, int], Dict[str, int]]:
    """
    Расчёт статистики данных.

    Args:
        fsm: файловый менеджер
        class_names: список имён классов

    Returns:
        * словарь выборок, где ключ имя выборки, а значение статистика по выборке
        * словарь форматов, где ключ это формат, а значение это количество
        * словарь цветовых режимов, где ключ это режим (RGB/L/...), а значение это количество

    Raises:
        VersionValidationError: если найдены повреждённые изображения
    """
    # Объекты для формирования метаданных
    splits: Dict[SplitType, Split] = {}
    image_format_stats: Dict[str, int] = {}
    color_mode_stats: Dict[str, int] = {}
    broken_files: List[str] = []

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
            # пополнение словарей форматов и цветовых режимов (счётчики суммируются между классами)
            for ext, cnt in image_statistics.format_stats.items():
                image_format_stats[ext] = image_format_stats.get(ext, 0) + cnt
            for mode, cnt in image_statistics.color_mode_stats.items():
                color_mode_stats[mode] = color_mode_stats.get(mode, 0) + cnt
            broken_files.extend(
                f"{selection}/{dir_class}/{name}" for name in image_statistics.broken_files
            )

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

    if broken_files:
        raise VersionValidationError(
            "Найдены повреждённые изображения (не открываются): " + ", ".join(broken_files)
        )

    logger.info(f'✅ Все папки проверены!')
    return splits, image_format_stats, color_mode_stats

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
    color_mode_stats: Dict[str, int] = field(default_factory=dict)
    broken_files: List[str] = field(default_factory=list)

def _class_statistic(
    fsm: FileSystemManager,
) -> ImageStatistics:
    """
    Сбор статистики по всем изображениям.
    Повреждённые изображения собираются в `broken_files` (решение об ошибке — на уровне выше).

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
            # Проверка целостности файла (verify портит объект — нужно открыть заново)
            with Image.open(img_path) as img:
                img.verify()

            # Статистика разрешений и цветовых режимов
            with Image.open(img_path) as img:
                size_str = f"{img.width}x{img.height}"
                image_statistics.size_counter[size_str] = image_statistics.size_counter.get(size_str, 0) + 1
                image_statistics.color_mode_stats[img.mode] = image_statistics.color_mode_stats.get(img.mode, 0) + 1

            # Статистика форматов
            ext = img_path.suffix.lower().lstrip('.')
            image_statistics.format_stats[ext] = image_statistics.format_stats.get(ext, 0) + 1
        except Exception as e:
            logger.warning(f"Повреждённое изображение {img_path.name}: {e}")
            image_statistics.broken_files.append(img_path.name)
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
