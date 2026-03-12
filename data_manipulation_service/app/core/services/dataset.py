import json
from typing import List
from pathlib import Path
from pydantic import ValidationError

from ..filesystem import FileSystemManager
from app.api.schemas.dataset import DatasetMetadata
from app.api.schemas.dataset_new import NewDataset

from logs import get_logger

logger = get_logger(__name__)
METADATA_DATASETS_NAME_FILE = 'metadata_ds.json'

class Dataset:
    """
    Класс для работы с датасетом.

    Основные возможности:
    - получение/изменение информации по датасету
    - Создание/получение/изменение/удаление версиями
    """

    def __init__(self):
        self._fsm = FileSystemManager()

    @property
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
            dataset_id: str,
            dsm: DatasetMetadata
    ) -> bool:
        """
        Сохранить метаданные в JSON-файл.
        """
        path = self._generate_meatadata_path(dataset_id)

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
                raise FileNotFoundError(f"Файл не найден: {path}")
            return path
        else: 
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with path.open('w', encoding="utf-8") as f:
                f.write('')
            return path

    def create_dataset_info(
            self, 
            dataset_id: str,
            dsm: DatasetMetadata
    ) -> bool:
        self._generate_meatadata_path(dataset_id, is_old_ds=False)
        return self.change_dataset_info(dataset_id, dsm)
    
    def create_new_dataset(
            self,
            dsn: NewDataset
    ) -> bool:
        # __WARNING__ рассматриваем только задачу с классификацией изображений
        # в будующем добавим разные задачи(регрессия/классификаия/...) и типы(текст/изображние/...) 
        if not (dsn.type == "image" and dsn.task == "classification"):
            raise TypeError(f"Тип {dsn.type} с задачей {dsn.task} не поддерживается")
        
        logger.info('Валидации датасета')
        self._validation_dataset_img_clf(dsn)

        return True
    
    def _validation_dataset_img_clf(self, dsn: NewDataset) -> bool:
        self._fsm.reset()
        self._fsm.in_dir('temp')

        if dsn.dataset_id not in self._fsm.get_all_dirs():
            raise FileNotFoundError(f"Не найден dataset с dataset_id = {dsn.dataset_id}")
        
        self._fsm.in_dir(dsn.dataset_id)

        dir_selections = {'train', 'test', 'val'}
        dir_actual_selections = set(self._fsm.get_all_dirs())

        if dir_actual_selections != dir_selections:
            raise ValueError(f"Ожидались ровно папки {dir_selections}, найдены: {dir_actual_selections}")

        for dir_selection in dir_selections:
            self._fsm.in_dir(dir_selection)
            dir_classes = self._fsm.get_all_dirs()

            if dir_classes != dsn.class_names:
                raise ValueError(f"В папке {dir_selection} отсутствует один из классов {dsn.class_names}")

            for dir_class in dir_classes:
                self._fsm.in_dir(dir_class)

                if not self._fsm.get_all_dirs():
                    raise ValueError(f'В папке {dir_selection}/{dir_class} есть лишние папки')

                if not self._fsm.all_file_is_image():
                    raise ValueError(f'В папке {dir_selection}/{dir_class} не все файлы являются изображениями')
                
                if not self._fsm.get_all_files():
                    raise ValueError(f'В папке {dir_selection}/{dir_class} нет файлов')

                self._fsm.out_dir()    

            self._fsm.out_dir()

        return True
        