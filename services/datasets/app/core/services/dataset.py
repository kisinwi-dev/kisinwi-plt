import json
from typing import List, Tuple, Dict
from pathlib import Path
from pydantic import ValidationError

from ..filesystem import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata, DatasetResponse, Version, VersionResponse
from app.api.schemas.dataset_new import NewDataset, NewVersion
from app.api.schemas.splits import (
    SplitSummaryResponse, SplitCountsResponse, SplitBalanceResponse,
    ClassDistributionResponse, ImageSizeStatsResponse
)
from app.api.schemas.comparison import (
    VersionComparisonResponse, CountsComparisonResponse,
    DistributionComparisonResponse, BalanceComparisonResponse,
    SizeStatsComparisonResponse, FilesDiffResponse, FilesDiffSummary
)
from app.core.exception.dataset import *
from app.core.exception.version import VersionNotFoundError, VersionComparisonError
from app.core.services.validation import (
    new_dataset as dvc,
    new_version as vvc
)
from . import comparison

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
        self._temp_path = (self._fsm.worker_path / 'temp').resolve()

    def get_datasets_id(self) -> List[str]:
        # __WARNING__ НА ДАННЫЙ МОМЕНТ РАССМАТРИВАЕТСЯ ВАРИАНТ, КОГДА У НАС ОДИН ПОЛЬЗОВАТЕЛЬ
        return [
            d for d in self._fsm.get_all_dirs()
            if (self._fsm.worker_path / d / METADATA_DATASETS_NAME_FILE).exists()
        ]
    
    # ================ (вспомогательные функции) получение данных из json ======================

    def _get_dataset_info(
        self, 
        dataset_id: str
    ) -> DatasetMetadata:
        """Загрузить метаданные датасета из JSON-файла"""
        path = self._generate_metadata_path(dataset_id)

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return DatasetMetadata.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Невалидный JSON в файле {path}: {e}")
        except ValidationError as e:
            raise ValueError(f"Структура метаданных некорректна: {e}")

    def _get_version_info(
        self,
        dataset_id: str,
        version_id: str
    ) -> Version:
        """Загрузить метаданные версии из JSON-файла""" 
        dsm = self._get_dataset_info(dataset_id)
        for version in dsm.versions:
            if version.id == version_id:
                return version
        raise VersionNotFoundError(version_id)
    
    # ================ проверки наличия данных ======================

    def dataset_exists(self, dataset_id: str) -> bool:
        return dataset_id in self.get_datasets_id()

    def version_exists(self, dataset_id: str, version_id: str) -> bool:
        versions = self._get_dataset_info(dataset_id).versions
        return any(v.id == version_id for v in versions)
    
    # ================ получение информации о разбиении ======================

    def get_version_split_summary(
        self,
        dataset_id: str,
        version_id: str
    ) -> SplitSummaryResponse:
        """Получение информации по распределениям"""
        version = self._get_version_info(dataset_id, version_id)
        return version.get_split_summary()

    def get_version_split_counts(
        self,
        dataset_id: str,
        version_id: str
    ) -> SplitCountsResponse:
        """Получение количества изображений по сплитам"""
        version = self._get_version_info(dataset_id, version_id)
        return version.get_split_counts()

    def get_version_split_balance(
        self,
        dataset_id: str,
        version_id: str
    ) -> SplitBalanceResponse:
        """Получение баланса классов по сплитам"""
        version = self._get_version_info(dataset_id, version_id)
        return version.get_split_balance()

    def get_version_class_distribution(
        self,
        dataset_id: str,
        version_id: str
    ) -> ClassDistributionResponse:
        """Получение распределения классов по сплитам"""
        version = self._get_version_info(dataset_id, version_id)
        return version.get_class_distribution_response()

    def get_version_image_size_stats(
        self,
        dataset_id: str,
        version_id: str
    ) -> ImageSizeStatsResponse:
        """Получение статистики размеров изображений по сплитам"""
        version = self._get_version_info(dataset_id, version_id)
        return version.get_image_size_stats()

    # ================ сравнение версий ======================

    def _get_versions_pair(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str
    ) -> Tuple[Version, Version]:
        """Загрузить пару версий для сравнения с валидацией параметров"""
        if from_version_id == to_version_id:
            raise VersionComparisonError("Нельзя сравнивать версию саму с собой")
        return (
            self._get_version_info(dataset_id, from_version_id),
            self._get_version_info(dataset_id, to_version_id)
        )

    def compare_versions(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str
    ) -> VersionComparisonResponse:
        """Полная сводка сравнения двух версий датасета"""
        from_v, to_v = self._get_versions_pair(dataset_id, from_version_id, to_version_id)
        files_diff = self.compare_version_files(dataset_id, from_version_id, to_version_id)
        return VersionComparisonResponse(
            dataset_id=dataset_id,
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            counts=comparison.compare_counts(dataset_id, from_v, to_v),
            distribution=comparison.compare_distribution(dataset_id, from_v, to_v),
            balance=comparison.compare_balance(dataset_id, from_v, to_v),
            size_stats=comparison.compare_size_stats(dataset_id, from_v, to_v),
            files=FilesDiffSummary(
                added_count=files_diff.added_count,
                removed_count=files_diff.removed_count,
                common_count=files_diff.common_count
            )
        )

    def compare_version_counts(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str
    ) -> CountsComparisonResponse:
        """Сравнение количества изображений по сплитам и классам"""
        from_v, to_v = self._get_versions_pair(dataset_id, from_version_id, to_version_id)
        return comparison.compare_counts(dataset_id, from_v, to_v)

    def compare_version_distribution(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str
    ) -> DistributionComparisonResponse:
        """Сравнение распределений классов (состав + drift-метрики)"""
        from_v, to_v = self._get_versions_pair(dataset_id, from_version_id, to_version_id)
        return comparison.compare_distribution(dataset_id, from_v, to_v)

    def compare_version_balance(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str
    ) -> BalanceComparisonResponse:
        """Сравнение баланса классов"""
        from_v, to_v = self._get_versions_pair(dataset_id, from_version_id, to_version_id)
        return comparison.compare_balance(dataset_id, from_v, to_v)

    def compare_version_size_stats(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str
    ) -> SizeStatsComparisonResponse:
        """Сравнение форматов и размеров изображений"""
        from_v, to_v = self._get_versions_pair(dataset_id, from_version_id, to_version_id)
        return comparison.compare_size_stats(dataset_id, from_v, to_v)

    def compare_version_files(
        self,
        dataset_id: str,
        from_version_id: str,
        to_version_id: str
    ) -> FilesDiffResponse:
        """По-файловый diff между версиями (по относительным путям)"""
        self._get_versions_pair(dataset_id, from_version_id, to_version_id)

        def _scan(version_id: str) -> set:
            with self._fsm.use_path(self._fsm.worker_path / dataset_id / version_id):
                return set(self._fsm.get_all_files_recursive())

        return comparison.build_files_diff(
            dataset_id,
            from_version_id,
            to_version_id,
            from_files=_scan(from_version_id),
            to_files=_scan(to_version_id)
        )

    # ================ выдача имеющейся информации о датасетах и версиях ======================

    def get_dataset_response_info(
        self, 
        dataset_id
    ) -> DatasetResponse:
        """Получение данных для выдачи информации по датасету"""
        dataset = self._get_dataset_info(dataset_id)
        return dataset.get_datasets_response()
        
    def get_version_info(
        self, 
        dataset_id: str,
        version_id: str
    ) -> VersionResponse:
        """Получение информации о версии датасета"""
        vm = self._get_version_info(dataset_id, version_id)
        return vm.get_version_response()

    def change_dataset_info(
        self, 
        dsm: DatasetMetadata
    ) -> bool:
        """
        Сохранить метаданные в JSON-файл.
        """
        path = self._generate_metadata_path(dsm.id)

        try:
            json_content = dsm.model_dump_json(indent=2)
            tmp = path.with_suffix(".tmp")

            with tmp.open("w", encoding="utf-8") as f:
                f.write(json_content)

            tmp.replace(path)

            return True
        except Exception as e:
            raise DatasetValidationError("", dsm.id)


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
        logger.debug(f"Создание нового датасета")

        # валидация датасета
        dsm = dvc(dsm_n)

        # перенос датасета из временных файлов в директорию датасетов
        self.promote_dataset(dsm,dsm_n)
        
        # сохранение метаданных датасета
        self._create_new_dataset_info(dsm.id, dsm)
        logger.debug(f'🟩 Создана новый датасет {dsm.id}')
        return True
    
    def add_new_version(
        self,
        dataset_id: str,
        v_n: NewVersion
    ) -> bool:
        """Создание новой версии для датасета"""
        logger.debug(f"Добавление версии в {dataset_id}")
        
        # получение данных о датасете
        dsm = self._get_dataset_info(dataset_id)
        
        # валидация данных
        v = vvc(dsm, v_n)
        
        # перенос полученных данных в папку датасета
        self.promote_version(v_n, dsm, v)

        # обновление метаданных датасета
        dsm.versions.append(v)
        self.change_dataset_info(dsm)

        logger.debug(f'🟩 Создана новая версия {v.id} для датасета {dsm.id}')
        return True
    
    # ================ удаление данных ======================
    
    def drop_version(
        self,
        dataset_id: str,
        version_id: str,
    ) -> bool:
        if dataset_id not in self.get_datasets_id():
            raise DatasetNotFoundError(dataset_id)
        
        dsm = self._get_dataset_info(dataset_id)

        if dsm.default_version_id == version_id:
            raise CannotDeleteDefaultVersion(dataset_id, version_id)

        new_list_versions = []
        dsm.versions = [v for v in dsm.versions if v.id != version_id]

        path_dataset = self._fsm.worker_path / dataset_id
        with self._fsm.use_path(path_dataset):

            if version_id not in self._fsm.get_all_dirs():
                raise VersionNotFoundError(version_id)

            self._fsm.delete(version_id)
        
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

    def drop_cache(
        self
    ):
        with self._fsm.use_path(self._temp_path):
            for dir in self._fsm.get_all_dirs():
                self._fsm.delete(dir)
        logger.warning('Кэш очищен')

    # ================ перенос датасета/версии в основную директорию из temp ======================

    def promote_dataset(
        self,
        dsm: DatasetMetadata,
        dsm_n: NewDataset
    ) -> None:
        """Перенос датасета из временных файлов в основную директорию"""
        path_dataset, new_path_version = self._generate_new_dataset_path(
            data_id=dsm_n.version.id_data,
            dataset_id=dsm.id,
            version_id=dsm.versions[0].id
        )
        
        with self._fsm.use_path(path_dataset):
            self._fsm.move_dir(new_path_version)

        with self._fsm.use_path(path_dataset.parent):
            self._fsm.delete(dsm_n.version.id_data)


    def promote_version(
        self,
        v_n: NewVersion,
        dsm: DatasetMetadata,
        v: Version
    ) -> None:
        """Перенос версии из временных файлов в датасет"""
        path_version, new_path_version = self._generate_new_version_path(
            data_id=v_n.id_data,
            dataset_id=dsm.id,
            version_id=v.id
        )
        
        with self._fsm.use_path(path_version):
            self._fsm.move_dir(new_path_version)

        with self._fsm.use_path(path_version.parent):
            self._fsm.delete(v_n.id_data)


    # ================ генерация path до нужных папок/файлов ======================

    def _generate_metadata_path(
        self, 
        dataset_id: str, 
        is_old_ds: bool = True
    ) -> Path:
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
        data_id: str,
        dataset_id: str,
        version_id: str
    ) -> Tuple[Path, Path]:
        """
        paths для нового dataset

        Args:
            data_id: id загруженных данных
            dataset_id: id датасета
            version_id: id новой версии

        Returns:
            * path_dataset - путь до данных из временной папки
            * new_path_version - путь до версии
        """
        path_dataset = (self._fsm.worker_path / 'temp' / data_id).resolve()
        new_path_dataset = (self._fsm.worker_path / dataset_id).resolve()
        new_path_version = (new_path_dataset / version_id).resolve()

        new_path_dataset.mkdir(parents=True, exist_ok=True) 
        new_path_version.mkdir(parents=True, exist_ok=True)
        
        return path_dataset, new_path_version

    def _generate_new_version_path(
        self,
        data_id: str,
        dataset_id: str,
        version_id: str
    ) -> Tuple[Path, Path]:
        """
        paths для новой версии datasets

        Args:
            data_id: id загруженных данных
            dataset_id: id датасета
            version_id: id новой версии

        Returns:
            * path_version - путь до данных из временной папки
            * new_path_version - путь до версии
        """
        path_version = (self._fsm.worker_path / 'temp' / data_id).resolve()
        new_path_version = (self._fsm.worker_path / dataset_id / version_id ).resolve()

        if not path_version.exists():
            raise FileNotFoundError(path_version)
        
        new_path_version.mkdir(parents=True, exist_ok=True)
        
        return path_version, new_path_version
