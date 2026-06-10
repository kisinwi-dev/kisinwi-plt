import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
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
        version_id: str,
        file: UploadFile
    ):
        """Сохранение файла"""
        try:
            # Санитизация имени: только basename, без обхода директорий
            safe_name = Path(file.filename or "").name
            if not safe_name or safe_name in (".", ".."):
                raise ValueError(f"Некорректное имя файла: {file.filename!r}")

            # переходим в папку по id версии и загружаем новые файлы
            version_dir = self._get_version_dir(version_id)

            file_path = version_dir / safe_name
            if file_path.exists():
                raise FileExistsError("Файл уже сущестует")

            # Сохраняем новые файлы
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # В БД храним путь относительно storage_dir (устойчив к переезду);
            # размер считаем из абсолютного пути
            rel_path = file_path.relative_to(self.storage_dir)
            file_size = file_path.stat().st_size
            self._add_info_file(
                version_id,
                safe_name,
                str(rel_path),
                file_size
            )

            logger.info(f"Сохранён файл: {safe_name} для версии '{version_id}'")
        except (FileExistsError, ValueError) as e:
            logger.error(f"Ошибка при сохранении нового файла '{file.filename}' для '{version_id}': {e}")
            raise e
        except Exception as e:
            logger.error(f"Ошибка при сохранении нового файла '{file.filename}' для '{version_id}': {e}")
            raise Exception(f"Не удалось сохранить файл {file.filename}: {e}")

    def get_info_files(
        self,
        version_id: str
    ) -> List[Dict[str, Any]] | None:
        """Получение информации о файлах конкретной версии"""
        query = f"""
            SELECT id, version_id, filename, file_size, created_at
            FROM {self._files_table}
            WHERE version_id = %s
        """

        with self.db as db:
            rows = db.fetch_all(
                query,
                (version_id,)
            )

        if len(rows) == 0:
            return None

        return [
            {
                "id": row[0],
                "version_id": row[1],
                "filename": row[2],
                "file_size": row[3],
                "created_at": row[4]
            }
            for row in rows
        ]

    def drop(
            self,
            version_id: str,
            id_files: List[str] | None = None
        ):
        """
        Удаление файлов по их id

        Args:
            version_id: ID версии модели
            id_files: Список ID файлов для удаления (если не указывать, удаляет все файлы)
        """
        version_dir = self._get_version_dir(version_id)

        if id_files:

            pl = ', '.join(['%s::uuid' for _ in id_files])
            query = f"""
                SELECT filename
                FROM ml_model_files
                WHERE version_id = %s AND id IN ({pl})
            """
            with self.db as db:
                files = db.fetch_all(query, (version_id, *id_files))
        else:
            query = """
                SELECT filename
                FROM ml_model_files
                WHERE version_id = %s
            """
            with self.db as db:
                files = db.fetch_all(query, (version_id,))

        for file_row in files:
            # путь реконструируем из storage_dir/version_id/filename
            file_path = version_dir / file_row[0]
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Удалён файл: {file_path}")

        deleted_count = self._delete_info(version_id, id_files)

        if not id_files and version_dir.exists() and not any(version_dir.iterdir()):
            version_dir.rmdir()
            logger.info(f"Удалена директория версии {version_id}")

        return deleted_count

    def _delete_info(
        self,
        version_id: str,
        file_ids: List[str] | None = None
    ) -> int:
        """
        Удаление из бд информации о файле/файлах

        Args:
            version_id: ID версии модели
            file_ids: Список ID файлов для удаления (если не указывать, удаляет все файлы)

        Returns:
            Количество удалённых записей
        """

        if file_ids: # Удаляем конкретные файлы

            pl = ', '.join(['%s::uuid' for _ in file_ids])
            query = f"""
                DELETE FROM ml_model_files
                WHERE version_id = %s AND id IN ({pl})
                RETURNING id
            """

            with self.db as db:
                result = db.fetch_all(query, (version_id, *file_ids))

                deleted_count = len(result)
                logger.info(f"Удалено {deleted_count} записей для версии {version_id} (конкретные файлы: {file_ids})")
                return deleted_count

        else: # Удаляем все файлы версии

            query = """
                DELETE FROM ml_model_files
                WHERE version_id = %s
                RETURNING id
            """

            with self.db as db:
                result = db.fetch_all(query, (version_id,))

                deleted_count = len(result)
                logger.info(f"Удалено {deleted_count} записей о файлах для версии {version_id}")
                return deleted_count

    def drop_version_dir(self, version_id: str) -> None:
        """
        Полностью удалить директорию версии с диска.

        Вызывается при удалении версии/модели: FK CASCADE убирает записи о файлах
        из БД, а физические файлы и директорию нужно удалить отдельно.
        """
        version_dir = self.storage_dir / str(version_id)
        if version_dir.exists():
            shutil.rmtree(version_dir, ignore_errors=True)
            logger.info(f"Удалена директория файлов версии {version_id}")

    def _get_version_dir(self, version_id: str) -> Path:
        """Получить путь к папке версии"""
        version_dir = self.storage_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    def _add_info_file(
        self,
        version_id: str,
        filename: str,
        file_path: str,
        file_size: int
    ):
        """Добавляем в БД информацию о файле (file_path — относительно storage_dir)"""
        query = """
            INSERT INTO ml_model_files (version_id, filename, file_path, file_size)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """

        with self.db as db:
            result = db.fetch_one(
                query,
                (version_id, str(filename), str(file_path), file_size)
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

    def get_file_path(self, file_id: str) -> Tuple[Path, str]:
        """Получить путь к файлу по его ID и его имя"""

        query = """
            SELECT version_id, filename
            FROM ml_model_files
            WHERE id = %s
        """

        with self.db as db:
            result = db.fetch_one(query, (file_id,))

            if not result:
                raise ValueError(f"Файл с ID {file_id} не найден")

            version_id, filename = result
            # путь реконструируем из storage_dir/version_id/filename
            file_path = self.storage_dir / str(version_id) / filename

            if not file_path.exists():
                raise ValueError(f"Физический файл не найден: {filename}")

            return file_path, filename
