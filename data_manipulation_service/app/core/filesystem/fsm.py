import shutil
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.png'}


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
    
    # ================ Получение информации об окружении ======================

    def status(self) -> str:
        """Относительный путь от корня до текущей папки"""
        try:
            rel = self.worker_path.relative_to(self._root)
            return str(rel) if rel != Path('.') else "."
        except ValueError:
            return "[ошибка пути]"

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

    def all_file_is_image(self, recursive: bool = False) -> bool:
        """
        Проверяет, что все файлы в папке являются изображениями
            recursive=True → проверяет и вложенные папки
        """
        def is_image(p: Path) -> bool:
            return p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES

        if recursive:
            return all(is_image(p) for p in self.worker_path.rglob("*") if p.is_file())
        else:
            return all(is_image(p) for p in self.worker_path.iterdir() if p.is_file())
