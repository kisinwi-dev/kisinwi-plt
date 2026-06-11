import asyncio
from collections import defaultdict
from typing import Dict, Set

from app.logs import get_logger

logger = get_logger(__name__)

class DiscussionStreamBroker:
    """In-memory pub/sub уведомлений об обновлении ленты дискуссии.

    Работает только в рамках одного процесса uvicorn (несколько воркеров
    разнесут издателя и подписчиков по разным процессам).
    Подписчик получает уведомление без данных и сам перечитывает
    актуальный снимок дискуссии из хранилища.
    """

    def __init__(self, max_queue_size: int = 32):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._max_queue_size = max_queue_size

    def subscribe(self, discussion_id: str) -> asyncio.Queue:
        """Очередь уведомлений об обновлениях дискуссии"""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[discussion_id].add(queue)
        logger.debug(f"Новый подписчик на дискуссию(id:{discussion_id})")
        return queue

    def unsubscribe(self, discussion_id: str, queue: asyncio.Queue):
        """Отписка; пустые наборы подписчиков удаляются"""
        subscribers = self._subscribers.get(discussion_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            del self._subscribers[discussion_id]
        logger.debug(f"Подписчик дискуссии(id:{discussion_id}) отключён")

    def publish(self, discussion_id: str):
        """Уведомление всех подписчиков дискуссии об её обновлении"""
        for queue in self._subscribers.get(discussion_id, ()):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                # Подписчик перечитает полный снимок по уже лежащему уведомлению
                pass

def format_sse(event: str, data: str) -> str:
    """Кадр Server-Sent Events: имя события и однострочный JSON в data"""
    return f"event: {event}\ndata: {data}\n\n"
