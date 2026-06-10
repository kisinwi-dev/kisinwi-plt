import os
import shutil
import tarfile
import time
import zipfile
from app.logs import get_logger
from pathlib import Path
from fastapi import UploadFile

logger = get_logger(__name__)

UNPACK_HANDLERS = {
    '.zip': '_unpack_zip',
    '.tar': '_unpack_tar',
    '.tar.gz': '_unpack_tar',
    '.tgz': '_unpack_tar',
}

class ArchiveManager:
    """
    Класс для работы с загруженными архивами:
    - сохранить загруженный файл
    - безопасно распаковать ZIP-архив
    - очистить всю временную папку
    """

    def __init__(self, temp_folder: Path = Path("datasets/temp")):
        """
        temp_folder — папка, куда будут сохраняться загруженные файлы 
        и распакованные архивы
        """
        self.temp_folder = temp_folder.resolve()
        if not self.temp_folder.is_dir():
            logger.warning(f"Папка временных файлов создана автоматически.")
            os.makedirs(temp_folder, exist_ok=True)

    def save_file(self, uploaded_file: UploadFile, name_file: str) -> Path:
        """Сохраняет загруженный файл с уникальным именем"""
        if not uploaded_file.filename:
            raise ValueError("У загруженного файла нет имени")

        save_path = self.temp_folder / name_file

        logger.info(f"Сохраняем файл: {name_file}")

        try:
            with save_path.open("wb") as f:
                shutil.copyfileobj(uploaded_file.file, f)
        except Exception as e:
            logger.error(f"Не удалось сохранить файл {uploaded_file.filename}: {e}")
            raise

        logger.info(f"Файл сохранён: {name_file}")
        return save_path

    def unpack(self, archive_path: Path, new_folder_name: str) -> Path:
        """
        Универсальная функция распаковки архива.
        В будущем можно легко добавить поддержку других форматов.
        """
        if not archive_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {archive_path}")

        # составные расширения (.tar.gz) проверяются раньше простых (.gz не поддерживается отдельно)
        name = archive_path.name.lower()
        suffix = next(
            (ext for ext in sorted(UNPACK_HANDLERS, key=len, reverse=True) if name.endswith(ext)),
            None
        )

        if suffix is None:
            supported = ", ".join(sorted(UNPACK_HANDLERS.keys()))
            raise ValueError(
                f"Формат архива не поддерживается: {archive_path.suffix}\n"
                f"Поддерживаемые форматы: {supported}"
            )

        extract_folder = self._create_temp_subfolder(new_folder_name)

        # вызываем соответствующий метод распаковки
        handler_name = UNPACK_HANDLERS[suffix]
        handler = getattr(self, handler_name)
        result = handler(archive_path, extract_folder)

        Path.unlink(archive_path)
        self._remove_junk_dirs(result)
        self._flatten_single_root(result)

        return result

    def _remove_junk_dirs(self, extract_folder: Path):
        """Удаляет служебные папки архиваторов (например __MACOSX от macOS)"""
        for junk in extract_folder.rglob("__MACOSX"):
            if junk.is_dir():
                logger.debug(f"Удалена служебная папка: {junk}")
                shutil.rmtree(junk)

    def _unpack_zip(self, zip_path: Path, extract_folder: Path) -> Path:
        """Распаковка ZIP-архива"""
        logger.info(f"Распаковываем ZIP: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                self._check_zip_is_safe(zf.namelist(), extract_folder)
                zf.extractall(extract_folder)
        except zipfile.BadZipFile:
            logger.error(f"Повреждённый ZIP: {zip_path}")
            raise ValueError("Некорректный или повреждённый ZIP-архив")
        except Exception as e:
            logger.error(f"Ошибка распаковки ZIP {zip_path}: {e}")
            raise

        logger.info(f"ZIP успешно распакован в: {extract_folder.name}")
        return extract_folder

    def _unpack_tar(self, tar_path: Path, extract_folder: Path) -> Path:
        """Распаковка TAR-архива (включая .tar.gz/.tgz)"""
        logger.info(f"Распаковываем TAR: {tar_path.name}")

        try:
            with tarfile.open(tar_path, "r:*") as tf:
                # filter='data' отбрасывает symlink'и, абсолютные пути и выход за пределы папки
                tf.extractall(extract_folder, filter="data")
        except tarfile.TarError as e:
            logger.error(f"Повреждённый TAR: {tar_path}: {e}")
            raise ValueError("Некорректный или повреждённый TAR-архив")
        except Exception as e:
            logger.error(f"Ошибка распаковки TAR {tar_path}: {e}")
            raise

        logger.info(f"TAR успешно распакован в: {extract_folder.name}")
        return extract_folder

    def _flatten_single_root(self, extract_folder: Path):
        """
        Если архив содержит единственную корневую папку (архив папки целиком),
        поднимает её содержимое на уровень extract_folder и удаляет обёртку
        """
        items = list(extract_folder.iterdir())
        if len(items) != 1 or not items[0].is_dir():
            return

        root_dir = items[0]
        logger.info(f"Архив содержит единственную папку `{root_dir.name}` — разворачиваем её содержимое")

        for item in root_dir.iterdir():
            shutil.move(str(item), extract_folder / item.name)
        root_dir.rmdir()

    def _create_temp_subfolder(self, folder_name: str) -> Path:
        """Создаёт уникальную подпапку внутри temp_folder"""
        folder_path = self.temp_folder / folder_name
        folder_path.mkdir(exist_ok=False)
        return folder_path

    def _check_zip_is_safe(self, members: list[str], target_folder: Path):
        """
        Защита от path traversal (zip slip)
        Проверяет, что все пути внутри архива остаются внутри целевой папки
        """
        target_resolved = target_folder.resolve()

        for member in members:
            if member.endswith("/"):
                continue  # пропускаем папки

            full_path = (target_folder / member).resolve()

            if not full_path.is_relative_to(target_resolved):
                logger.error(f"Обнаружена попытка выхода за пределы: {member}")
                raise PermissionError(f"Небезопасный архив: попытка path traversal ({member})")

    def cleanup_stale(self, ttl_hours: float = 24) -> int:
        """
        Удаляет содержимое временной папки старше ttl_hours (по mtime).
        Возвращает количество удалённых элементов.
        """
        deadline = time.time() - ttl_hours * 3600
        removed = 0

        for item in self.temp_folder.iterdir():
            try:
                if item.stat().st_mtime > deadline:
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                removed += 1
                logger.info(f"Удалены устаревшие временные данные: {item.name}")
            except Exception as e:
                logger.error(f"Не удалось удалить {item}: {e}")

        return removed

    def clear_temp_folder(self):
        """
        Удаляет всё содержимое временной папки
        """
        logger.warning(f"Очистка всей папки: {self.temp_folder}")

        for item in self.temp_folder.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    logger.debug(f"Удалена папка: {item.name}")
                elif item.is_file():
                    item.unlink()
                    logger.debug(f"Удалён файл: {item.name}")
            except Exception as e:
                logger.error(f"Не удалось удалить {item}: {e}")

        logger.info("Временная папка очищена")