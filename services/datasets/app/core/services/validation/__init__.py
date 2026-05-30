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

        dsm = processor(dsn)

        logger.debug(f'🏁 Валидация пройдена')
        return dsm

def new_version(
    dsm: DatasetMetadata,
    new_version: NewVersion,
) -> Version:
    """Валидация при добавлении новой версии в существующий датасет"""
    # __WARNING__ 
    # рассматриваем только задачи с классификацией изображений 
    # в будущем добавим разные задачи(регрессия/классификаия/...) и типы(текст/изображние/...) 

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

    version = processor(new_version, dsm)

    logger.debug(f'🏁 Валидация пройдена')
    return version