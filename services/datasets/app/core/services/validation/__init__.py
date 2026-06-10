from typing import Dict, Tuple

from .registry import PREPROC_LOADERS_DATASET, PREPROC_LOADERS_VERSION, type_task_supported

from app.core.filesystem.fsm import FileSystemManager
from app.core.exception.dataset import (
    UnsupportedDatasetError
)
from app.api.schemas.dataset import DatasetMetadata, Version
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.logs import get_logger

logger = get_logger(__name__)

def new_dataset(
    dsn: NewDataset
) -> Tuple[DatasetMetadata, Dict[str, str]]:
        """Валидация при созданиии нового датасета данных. Возвращает метаданные и карту хешей файлов"""
        logger.info('⬜ Валидация...')

        processor = PREPROC_LOADERS_DATASET.get((dsn.type, dsn.task))

        logger.debug(f'Проверка на поддержку заданного типа данных и задачу...')
        if processor is None:
            raise UnsupportedDatasetError(
                dsn.task,
                dsn.type,
                type_task_supported()
            )
        logger.debug(f'✅ Тип:    {dsn.type}')
        logger.debug(f'✅ Задача: {dsn.task}')

        # настройка fsm
        fsm = _fsm_setting(dsn.version.id_data)

        # процесс валидации
        dsm, hashes = processor(fsm, dsn)

        logger.debug(f'🏁 Валидация пройдена')
        return dsm, hashes

def new_version(
    dsm: DatasetMetadata,
    nv: NewVersion,
) -> Tuple[Version, Dict[str, str]]:
    """Валидация при добавлении новой версии в существующий датасет. Возвращает метаданные версии и карту хешей файлов"""
    logger.info('⬜ Валидация...')

    processor = PREPROC_LOADERS_VERSION.get((dsm.type, dsm.task))

    logger.debug(f'Проверка на поддержку заданного типа данных и задачу...')
    if processor is None:
        raise UnsupportedDatasetError(
            dsm.task,
            dsm.type,
            type_task_supported()
        )
    logger.debug(f'✅ Тип:    {dsm.type}')
    logger.debug(f'✅ Задача: {dsm.task}')

    # настройка fsm 
    fsm = _fsm_setting(nv.id_data)

    # процесс валидации
    version, hashes = processor(fsm, nv, dsm)

    logger.debug(f'🏁 Валидация пройдена')
    return version, hashes

def _fsm_setting(
    id_data: str
) -> FileSystemManager:
    """
    Выдаёт настроенный файловый менеджер.
    * Настроенный fsm находится в директории загруженных данных.
    """
    fsm = FileSystemManager()
    fsm.in_dir("temp")

    if id_data not in fsm.get_all_dirs():
        logger.debug(f"🟥 Не найдены данные `{id_data}` по пути '{fsm.worker_path}'")
        raise FileNotFoundError(f"В папке temp не найдена директория '{id_data}'")
    fsm.in_dir(id_data)
    return fsm