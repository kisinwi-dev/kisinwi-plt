import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile

from .postresql import PostgresManager
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)

class FilesManager:
    def __init__(self, storage_dir: str = "model_files"):
        self.db = PostgresManager(postgresql_config.URL)
        self._files_table = "ml_model_files"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
       
    def add_file(
        self,
        model_id: str,
        file: UploadFile
    ):
        """Сохранение файла"""
        try:
            # переходим в папку по id модели и загружаем новые файлы
            model_dir = self._get_model_dir(model_id)

            file_path = model_dir / str(file.filename)
            if file_path.exists():
                raise FileExistsError("Файл уже сущестует")
            
            # Сохраняем новые файлы
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            self._add_info_file(
                model_id, 
                str(file.filename), 
                file_path
            )

            logger.info(f"Сохранён файл: {file.filename} для модели '{model_id}'")
        except FileExistsError as e:
            logger.error(f"Ошибка при сохранении нового файла '{file.filename}' для '{model_id}': {e}")
            raise e
        except Exception as e:
            logger.error(f"Ошибка при сохранении нового файла '{file.filename}' для '{model_id}': {e}")
            raise Exception(f"Не удалось сохранить файл {file.filename}: {e}")

    def get_info_files(
        self, 
        model_id: str
    ) -> List[Dict[str, Any]] | None:
        """Получение информации о файлах конретной модели"""
        query = f"""
            SELECT id, model_id, filename, file_size, created_at
            FROM {self._files_table}
            WHERE model_id = %s
        """

        with self.db as db:    
            rows = db.fetch_all(
                query,
                (model_id,)
            )
        
        if len(rows) == 0:
            return None
        
        return [
            {
                "id": row[0],
                "model_id": row[1],
                "filename": row[2],
                "file_size": row[3],
                "created_at": row[4]
            }
            for row in rows
        ]

    def _get_model_dir(self, model_id: str) -> Path:
        """Получить путь к папке модели"""
        model_dir = self.storage_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _add_info_file(
        self, 
        model_id: str,
        filename: str,
        file_path: Path
    ):
        """Добавляем в БД информацию о файле"""
        query = """
            INSERT INTO ml_model_files (model_id, filename, file_path, file_size)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """

        file_size = file_path.stat().st_size

        with self.db as db:
            result = db.fetch_one(
                query,
                (model_id, str(filename), str(file_path), file_size)
            )
            
            if not result:
                raise RuntimeError(f"Не удалось создать запись для файла {filename}")
            
            file_id = str(result[0])
            logger.info(f"Добавлена информация о файле '{file_id}': {filename}")

    def _clear_directory(self, directory: Path):
        """Полная очистка папки"""
        if directory.exists():
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            logger.debug(f"Очищена: {directory}")
