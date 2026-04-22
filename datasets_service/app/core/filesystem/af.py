import uuid
import shutil
import zipfile
import logging
from pathlib import Path
from fastapi import UploadFile

logger = logging.getLogger(__name__)

# Предволагается, что мы расширим список принимаемых расширений архивов
UNPACK_HANDLERS = {
    '.zip': '_unpack_zip',
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
            raise NotADirectoryError(f"Папка не существует: {self.temp_folder}")

    def save_file(self, uploaded_file: UploadFile, name_file: str) -> Path:
        """Сохраняет загруженный файл с уникальным именем"""
        if not uploaded_file.filename:
            raise ValueError("У загруженного файла нет имени")

        save_path = self.temp_folder / name_file

        logger.debug(f"Сохраняем файл: {save_path}")

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

        suffix = archive_path.suffix.lower()

        if suffix not in UNPACK_HANDLERS:
            supported = ", ".join(sorted(UNPACK_HANDLERS.keys()))
            raise ValueError(
                f"Формат архива не поддерживается: {suffix}\n"
                f"Поддерживаемые форматы: {supported}"
            )

        extract_folder = self._create_temp_subfolder(new_folder_name)

        # вызываем соответствующий метод распаковки
        handler_name = UNPACK_HANDLERS[suffix]
        handler = getattr(self, handler_name)
        result = handler(archive_path, extract_folder)

        Path.unlink(archive_path)

        return result 

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