import threading

from app.core.memory import discussion_context
from app.logs import get_logger

logger = get_logger(__name__)


class PipelineCancelled(Exception):
    """Пайплайн остановлен пользователем (это не сбой)."""


class CancellationRegistry:
    """
    Реестр флагов отмены пайплайнов, общий между потоками.

    Запрос на остановку приходит из отдельного HTTP-запроса (свой поток), а
    проверка флага идёт в фоновом потоке пайплайна — поэтому contextvars не
    подходят (они потоко-локальны). Храним модульный dict из threading.Event
    под Lock, ключ — discussion_id.
    """

    def __init__(self) -> None:
        self._flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def register(self, discussion_id: str) -> None:
        """Зарегистрировать пайплайн как запускаемый (создать флаг)."""
        with self._lock:
            self._flags[discussion_id] = threading.Event()

    def request_stop(self, discussion_id: str) -> bool:
        """
        Запросить остановку пайплайна.

        Returns:
            True, если пайплайн зарегистрирован (остановка принята),
            False — если такого активного пайплайна нет.
        """
        with self._lock:
            event = self._flags.get(discussion_id)
        if event is None:
            return False
        event.set()
        logger.info(f"CancellationRegistry: запрошена остановка discussion_id={discussion_id}")
        return True

    def is_stop_requested(self, discussion_id: str) -> bool:
        with self._lock:
            event = self._flags.get(discussion_id)
        return event is not None and event.is_set()

    def discard(self, discussion_id: str) -> None:
        """Убрать флаг после завершения пайплайна."""
        with self._lock:
            self._flags.pop(discussion_id, None)


cancellation_registry = CancellationRegistry()


def raise_if_cancelled() -> None:
    """
    Бросить PipelineCancelled, если для текущей дискуссии запрошена остановка.

    discussion_id берётся из discussion_context (установлен в фоновом раннере).
    """
    if not discussion_context.is_set():
        return
    discussion_id = discussion_context.get()
    if cancellation_registry.is_stop_requested(discussion_id):
        raise PipelineCancelled(f"Пайплайн дискуссии {discussion_id} остановлен пользователем")
