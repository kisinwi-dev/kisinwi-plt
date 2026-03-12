from .registry import PREPROC_LOADERS, type_task_supported

from app.core.filesystem.fsm import FileSystemManager
from app.core.exception.dataset import (
     DatasetAlreadyExistsError,
     UnsupportedDatasetError
)
from app.api.schemas.dataset import DatasetMetadata
from app.api.schemas.dataset_new import NewDataset
from app.logs import get_logger

logger = get_logger(__name__)

def new_dataset(
            dsn: NewDataset, 
            fsm: FileSystemManager | None = None
    ) -> DatasetMetadata:
        """Валидация полученных данных"""
        # __WARNING__ 
        # рассматриваем только задачи с классификацией изображений 
        # в будущем добавим разные задачи(регрессия/классификаия/...) и типы(текст/изображние/...) 

        logger.info('⬜ Валидация...')

        if fsm is None:
            fsm = FileSystemManager()
            fsm.in_dir('temp')
        
        # датасет существует
        fsm_ds = FileSystemManager()
        if dsn.dataset_id in fsm_ds.get_all_dirs():
            raise DatasetAlreadyExistsError(dsn.dataset_id)
        del fsm_ds
        
        processor = PREPROC_LOADERS.get((dsn.type, dsn.task))

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