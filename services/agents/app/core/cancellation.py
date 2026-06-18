import threading
from multiprocessing import Process
from typing import Optional

from app.logs import get_logger

logger = get_logger(__name__)


class ProcessRegistry:
    """
    Реестр запущенных пайплайнов: discussion_id → дочерний Process.

    Пайплайн выполняется в отдельном процессе, чтобы отмену можно было сделать
    мгновенной — просто убить процесс (рвутся сокеты текущего LLM-запроса, нет
    ретраев и последующих вызовов). Запрос на остановку приходит из отдельного
    HTTP-потока, поэтому доступ к реестру под Lock.
    """

    def __init__(self) -> None:
        self._procs: dict[str, Process] = {}
        self._lock = threading.Lock()

    def register(self, discussion_id: str, proc: Process) -> None:
        with self._lock:
            self._procs[discussion_id] = proc

    def get(self, discussion_id: str) -> Optional[Process]:
        with self._lock:
            return self._procs.get(discussion_id)

    def discard(self, discussion_id: str) -> None:
        """Убрать процесс из реестра после его завершения."""
        with self._lock:
            self._procs.pop(discussion_id, None)


process_registry = ProcessRegistry()
