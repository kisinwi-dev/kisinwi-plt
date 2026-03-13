import json
from typing import List
from pathlib import Path
from pydantic import ValidationError

from ..filesystem import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.exception.dataset import *
from app.core.services.validation import (
    new_dataset as dvc,
    new_version as vvc
)

from app.logs import get_logger

logger = get_logger(__name__)
METADATA_DATASETS_NAME_FILE = 'metadata_ds.json'

class DatasetManager:
    """
    Класс для работы с датасетом.

    Основные возможности:
    - получение/изменение информации по датасету
    - Создание/получение/изменение/удаление версиями
    """

    def __init__(self):
        self._fsm = FileSystemManager()

    def get_datasets_id(self) -> List[str]:
        # __WARNING__ НА ДАННЫЙ МОМЕНТ РАССМАТРИВАЕТСЯ ВАРИАНТ, КОГДА У НАС ОДИН ПОЛЬЗОВАТЕЛЬ
        self._fsm.reset()
        return self._fsm.get_all_dirs()
    
    def get_dataset_info(self, dataset_id) -> DatasetMetadata:
        """Загрузить метаданные из JSON-файла"""
        
        path = self._generate_meatadata_path(dataset_id)

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return DatasetMetadata.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Невалидный JSON в файле {path}: {e}")
        except ValidationError as e:
            raise ValueError(f"Структура метаданных некорректна: {e}")
    

    def change_dataset_info(
            self, 
            dsm: DatasetMetadata
    ) -> bool:
        """
        Сохранить метаданные в JSON-файл.
        """
        path = self._generate_meatadata_path(dsm.dataset_id)

        try:
            json_content = dsm.model_dump_json(indent=2)

            with path.open('w', encoding="utf-8") as f:
                f.write(json_content)

            return True
        except Exception as e:
            raise MemoryError("")

    def _generate_meatadata_path(self, dataset_id: str, is_old_ds: bool = True) -> Path:
        path = (self._fsm.worker_path / dataset_id / METADATA_DATASETS_NAME_FILE).resolve()
        if is_old_ds:
            if not path.is_file():
                raise DatasetNotFoundError(dataset_id)
            return path
        else: 
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with path.open('w', encoding="utf-8") as f:
                f.write('')
            return path

    def _create_dataset_info(
            self, 
            dataset_id: str,
            dsm: DatasetMetadata
    ) -> bool:
        self._generate_meatadata_path(dataset_id, is_old_ds=False)
        return self.change_dataset_info(dsm)

    def add_new_dataset(
            self,
            dsm_n: NewDataset
    ) -> bool:
        """Создание нового датасета"""
        dsm = dvc(dsm_n)

        new_path_dataset = self._fsm.worker_path
        self._fsm.in_dirs(['temp', dsm.dataset_id])
        self._fsm.move_dir(new_path_dataset)

        self._create_dataset_info(dsm.dataset_id, dsm)
        self._fsm.reset()
        return True
    
    def add_new_version(
            self,
            dataset_id: str,
            new_version: NewVersion
    ) -> bool:
        """Создание новой версии для датасета"""
        dsm = self.get_dataset_info(dataset_id)
        
        version = vvc(dsm, new_version)
        dsm.versions.append(version)
        self.change_dataset_info(dsm)
        
        self._fsm.in_dirs([dataset_id])
        new_path_dataset = self._fsm.worker_path
        self._fsm.reset()
        self._fsm.in_dirs(['temp', version.version_id])
        self._fsm.move_dir(new_path_dataset)

        return True
    
