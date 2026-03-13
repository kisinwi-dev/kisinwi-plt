import json
from typing import List, Tuple
from pathlib import Path
from pydantic import ValidationError

from ..filesystem import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.core.exception.dataset import *
from app.core.exception.version import VersionNotFoundError
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
        return [
            d for d in self._fsm.get_all_dirs()
            if (self._fsm.worker_path / d / METADATA_DATASETS_NAME_FILE).exists()
        ]
    
    # ================ проверки наличия данных ======================

    def dataset_exists(self, dataset_id: str) -> bool:
        return dataset_id in self.get_datasets_id()
    
    def version_exists(self, dataset_id: str, version_id: str) -> bool:
        versions = self.get_dataset_info(dataset_id).versions
        return any(v.version_id == version_id for v in versions)
    
    def get_dataset_info(self, dataset_id) -> DatasetMetadata:
        """Загрузить метаданные из JSON-файла"""
        
        path = self._generate_metadata_path(dataset_id)

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
        path = self._generate_metadata_path(dsm.dataset_id)

        try:
            json_content = dsm.model_dump_json(indent=2)
            tmp = path.with_suffix(".tmp")

            with tmp.open("w", encoding="utf-8") as f:
                f.write(json_content)

            tmp.replace(path)

            return True
        except Exception as e:
            raise DatasetValidationError("", dsm.dataset_id)


    def _create_new_dataset_info(
            self, 
            dataset_id: str,
            dsm: DatasetMetadata
    ) -> bool:
        self._generate_metadata_path(dataset_id, is_old_ds=False)
        return self.change_dataset_info(dsm)

    # ================ добавление новых данных ======================

    def add_new_dataset(
            self,
            dsm_n: NewDataset
    ) -> bool:
        """Создание нового датасета"""
        dsm = dvc(dsm_n)

        path_dataset, _, new_path_version = self._generate_new_dataset_path(dsm_n)
        
        self._fsm.set_path_worker(path_dataset)
        self._fsm.move_dir(new_path_version)
        self._fsm.reset()

        self._create_new_dataset_info(dsm.dataset_id, dsm)
        logger.debug(f'🟩 Создан новый датасет: {dsm.dataset_id}')
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
        
        path_version, new_path_version = self._generate_new_version_path(dsm.dataset_id, version.version_id)
        
        self._fsm.set_path_worker(path_version)
        self._fsm.move_dir(new_path_version)
        self._fsm.reset()
        logger.debug(f'🟩 Создана новая версия {version.version_id} для датасета {dsm.dataset_id}')
        return True
    
    # ================ удаление данных ======================
    
    def drop_version(
            self,
            dataset_id: str,
            version_id: str,
    ) -> bool:
        if dataset_id not in self.get_datasets_id():
            raise DatasetNotFoundError(dataset_id)
        
        dsm = self.get_dataset_info(dataset_id)

        if dsm.default_version_id == version_id:
            raise CannotDeleteDefaultVersion(dataset_id, version_id)

        new_list_versions = []
        dsm.versions = [v for v in dsm.versions if v.version_id != version_id]

        try:
            self._fsm.in_dir(dataset_id)
            if version_id not in self._fsm.get_all_dirs():
                raise VersionNotFoundError(version_id)
            
            self._fsm.delete(version_id)
        finally:
            self._fsm.reset()
        
        self.change_dataset_info(dsm)
        logger.info(f'Версия {version_id} удалена из датасета {dataset_id}')
        return True

    def drop_dataset(
            self, 
            dataset_id: str
    ) -> bool:
        if dataset_id not in self.get_datasets_id():
            raise DatasetNotFoundError(dataset_id)
        
        self._fsm.delete(dataset_id)
        logger.info(f'Датасета {dataset_id} удалён')
        return True
            

    # ================ генерация path до нужных папок/файлов ======================

    def _generate_metadata_path(self, dataset_id: str, is_old_ds: bool = True) -> Path:
        """
        Генерируем путь до метаданных
        
        Args:
            is_old_ds = True проверяет существование файла метаданных
        """
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
        
    def _generate_new_dataset_path(
            self, 
            dsm: NewDataset
    ) -> Tuple[Path, Path, Path]:
        """
        paths для нового dataset

        Returns:
            * path_dataset - путь до данных из временной папки
            * new_path_dataset - путь до датасета
            * new_path_version - путь до версии
        """
        path_dataset = (self._fsm.worker_path / 'temp' / dsm.dataset_id).resolve()
        new_path_dataset = (self._fsm.worker_path / dsm.dataset_id).resolve()
        new_path_version = (new_path_dataset / dsm.version.version_id ).resolve()

        new_path_dataset.mkdir(parents=True, exist_ok=True)
        new_path_version.mkdir(parents=True, exist_ok=True)
        
        return path_dataset, new_path_dataset, new_path_version

    def _generate_new_version_path(
            self,
            datset_id: str,
            version_id: str
    ) -> Tuple[Path, Path]:
        """
        paths для новой версии datasets

        Returns:
            * path_version - путь до данных из временной папки
            * new_path_version - путь до версии
        """
        path_version = (self._fsm.worker_path / 'temp' / version_id).resolve()
        new_path_version = (self._fsm.worker_path / datset_id / version_id ).resolve()

        if not path_version.exists():
            raise FileNotFoundError(path_version)
        
        new_path_version.mkdir(parents=True, exist_ok=True)
        
        return path_version, new_path_version
    