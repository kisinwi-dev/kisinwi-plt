from .registry import PREPROC_LOADERS_DATASET, PREPROC_LOADERS_VERSION, type_task_supported

from app.core.filesystem.fsm import FileSystemManager
from app.core.exception.dataset import (
     DatasetAlreadyExistsError, DatasetNotFoundError,
     UnsupportedDatasetError
)
from app.core.exception.version import VersionNotFoundError
from app.api.schemas.dataset import DatasetMetadata, Version
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.logs import get_logger

logger = get_logger(__name__)

def new_dataset(
            dsn: NewDataset, 
    ) -> DatasetMetadata:
        """Валидация при созданиии нового датасета данных"""
        # __WARNING__ 
        # рассматриваем только задачи с классификацией изображений 
        # в будущем добавим разные задачи(регрессия/классификаия/...) и типы(текст/изображние/...) 

        logger.info('⬜ Валидация...')

        
        fsm = FileSystemManager()
        fsm.in_dir('temp')
        
        # проверка на существование датасета
        fsm_ds = FileSystemManager()
        if dsn.dataset_id in fsm_ds.get_all_dirs():
            raise DatasetAlreadyExistsError(dsn.dataset_id)
        del fsm_ds
        
        processor = PREPROC_LOADERS_DATASET.get((dsn.type, dsn.task))

        if processor is None:
            raise UnsupportedDatasetError(
                dsn.task,
                dsn.type,
                type_task_supported()
            )

        logger.debug(
            f'| Поддержка тип и задачи'
            f'| 🟩 Тип:    {dsn.type}'
            f'| 🟩 Задача: {dsn.task}'
        )

        dsm = processor(dsn, fsm)

        logger.debug(f'🏁 Валидация пройдена')
        return dsm

def new_version(
    dsm: DatasetMetadata,
    new_version: NewVersion,
    fsm: FileSystemManager | None = None
) -> Version:
    """Валидация при добавлении новой версии в существующий датасет"""
    # __WARNING__ 
    # рассматриваем только задачи с классификацией изображений 
    # в будущем добавим разные задачи(регрессия/классификаия/...) и типы(текст/изображние/...) 

    logger.info('⬜ Валидация...')

    
    fsm = FileSystemManager()
    fsm.in_dir('temp')
    
    # проверка на существование датасета
    fsm_ds = FileSystemManager()
    if dsm.dataset_id not in fsm_ds.get_all_dirs():
        raise DatasetNotFoundError(dsm.dataset_id)
    
    # проверка наличия версии в temp
    if new_version.version_id not in fsm.get_all_dirs():
        raise VersionNotFoundError(new_version.version_id)
    
    fsm.in_dir(new_version.version_id)

    processor = PREPROC_LOADERS_VERSION.get((dsm.type, dsm.task))

    if processor is None:
        raise UnsupportedDatasetError(
            dsm.task,
            dsm.type,
            type_task_supported()
        )

    logger.debug(f'| Поддержка тип и задачи')
    logger.debug(f'| 🟩 Тип:    {dsm.type}')
    logger.debug(f'| 🟩 Задача: {dsm.task}')

    version = processor(dsm.class_names, new_version, fsm)

    logger.debug(f'🏁 Валидация пройдена')
    return version