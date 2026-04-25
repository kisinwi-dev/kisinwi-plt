import shutil
from typing import List
from contextlib import contextmanager
from pathlib import Path

from app.core.exception.version import VersionValidationError
from app.logs import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = {'.jpg', '.png', '.jpeg'}

class FileSystemManager:
    def __init__(
            self,
            root: Path | None = None
    ):
        """
        Файловый менеджер отвечает за взаимодействие файловой системой:
            * зайти в папку
            * выйти на уровень выше
            * удалить файл/папку 
            * вывести все имеющиеся файлы/папки
        """
        self._root = (root or Path.cwd() / "datasets").resolve()
        if not self._root.is_dir():
            raise NotADirectoryError(f"Корневая папка не найдена: {self._root}")
        self.worker_path = self._root

    # ================ Расположение в системе ======================

    @contextmanager
    def use_path(self, path: Path):
        """
        Временно установить рабочую директорию.
        После выхода из блока автоматически возвращается предыдущая.
        """
        new_path = path.resolve()

        if not new_path.exists():
            raise FileNotFoundError(new_path)

        if not new_path.is_relative_to(self._root):
            raise PermissionError("Нельзя выйти за пределы корневой директории")

        old_path = self.worker_path
        self.worker_path = new_path

        try:
            yield
        finally:
            self.worker_path = old_path

    def in_dir(self, dir_name: str) -> None:
        """
        Перейти в папку
        """
        new_path = (self.worker_path / dir_name).resolve()

        if not new_path.is_dir():
            raise FileNotFoundError(f"Папка не найдена: {dir_name}")

        if not new_path.is_relative_to(self._root):
            raise PermissionError("Нельзя выйти за пределы корневой директории")

        self.worker_path = new_path

    def set_path_worker(self, path: Path):
        """Заменяем рабочую директорию"""
        self.worker_path = path.resolve()

    def in_dirs(
            self,
            dirs_list: list[str]
    ) -> None:
        """
        Перейти в папки где:
            * каждый следующий элемент в списке является подпапкой предыдущего
        """
        for dir in dirs_list:
            self.in_dir(dir)

    def out_dir(self) -> None:
        """
        Выйти на уровень выше
        """
        if self.worker_path == self._root:
            raise PermissionError("Нельзя выйти за пределы корневой директории")

        self.worker_path = self.worker_path.parent

    def reset(self):
        """
        Вернуться в корень
        """
        self.worker_path = self._root
    
    # ================ Перемещение папки ======================

    def move_dir(self, new_path: Path) -> bool:
        """
        Перемещает всё содержимое текущей папки в new_path и переходит в новую папку
        """
        new_path = new_path.resolve()

        if not new_path.is_relative_to(self._root):
            raise PermissionError("Нельзя выйти за пределы корневой директории")

        if not new_path.exists():
            raise FileExistsError(f"Папка не существует: {new_path}")

        for item in self.worker_path.iterdir():
            if item == new_path:
                continue
            shutil.move(str(item), new_path)

        self.worker_path = new_path
        return True

    # ================ Получение информации об окружении ======================

    def status(self) -> str:
        """Относительный путь от корня до текущей папки"""
        try:
            rel = self.worker_path.relative_to(self._root)
            return str(rel) if rel != Path('.') else "."
        except ValueError:
            return "[ошибка пути]"

    def get_all(self) -> list[str]:
        "Возвращает всё что есть в текущем path"
        return [path.name for path in self.worker_path.iterdir()]

    def get_all_dirs(self) -> list[str]:
        return [path.name for path in self.worker_path.iterdir() if path.is_dir()]

    def get_all_files(self) -> list[str]:
        return [path.name for path in self.worker_path.iterdir() if path.is_file()]

    # ================ Переименовывание файлов ======================

    def rename_dir(
            self,
            name: str,
            new_name: str
    ):
        self._check_dir_exists(name)  
        self._rename_obj(name, new_name)

    def rename_file(
            self,
            name: str,
            new_name: str,
    ):
        self._check_file_exists(name)
        self._rename_obj(name, new_name)

    def _rename_obj(
            self,
            name: str,
            new_name: str
    ):
        """
        Переименовать объект
        """
        if not new_name or new_name in {".", ".."} or "/" in new_name or "\\" in new_name:
            raise ValueError(f"Недопустимое новое имя: {new_name!r}")
        
        old_path = self.worker_path / name
        new_path = self.worker_path / new_name

        if new_path.exists():
            raise FileExistsError(f"Обьект '{new_name}' уже существует")
        
        old_path.rename(new_path)

    # ================ Наличие обьекта ======================

    def _check_dir_exists(self, name: str) -> None:
        if name not in self.get_all_dirs():
            raise FileNotFoundError(f"Папка не найдена: {name}")

    def _check_file_exists(self, name: str) -> None:
        if name not in self.get_all_files():
            raise FileNotFoundError(f"Файл не найден: {name}")

    # ================ Переименовывание файлов ======================

    def delete(self, name: str) -> None:
        """
        Удалить файл или папку по имени в текущей директории
        """
        path = self.worker_path / name
        if not path.exists():
            raise FileNotFoundError(f"Не найден: {name}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    # ================ Работа с изображениями ================

    def all_file_is_image(
            self
        ) -> List[Path]:
        """
        Проверяет, что все файлы в папке являются изображениями
        
        Args:
            info - в какой директории проверяются файлы
        """
        def is_image(p: Path) -> bool:
            return p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES

        ps = []
        for p in self.worker_path.rglob("*"):
            if p.is_file(): 
                if is_image(p):
                    ps.append(p)
                else:
                    raise VersionValidationError(
                        f"В директории {self.worker_path} файл {p.name} не является изображениям.",
                    )
        return ps

    # ================ Размер папки ======================

    def get_dir_size(self, recursive: bool = True) -> int:
        """
        Возвращает размер текущей папки в байтах.
        """
        total = 0

        if recursive:
            for path in self.worker_path.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        else:
            for path in self.worker_path.iterdir():
                if path.is_file():
                    total += path.stat().st_size

        return total
