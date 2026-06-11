import asyncio
from collections import defaultdict
from typing import Dict, Set

from app.logs import get_logger

logger = get_logger(__name__)

class MetricStreamBroker:
    """In-memory pub/sub уведомлений об обновлении метрик модели.

    Работает только в рамках одного процесса uvicorn (несколько воркеров
    разнесут издателя и подписчиков по разным процессам).
    Подписчик получает уведомление без данных и сам перечитывает
    актуальный снимок метрик из БД.
    """

    def __init__(self, max_queue_size: int = 32):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._max_queue_size = max_queue_size

    def subscribe(self, model_id: str) -> asyncio.Queue:
        """Очередь уведомлений об обновлениях метрик модели"""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers[model_id].add(queue)
        logger.debug(f"Новый подписчик на метрики модели(id:{model_id})")
        return queue

    def unsubscribe(self, model_id: str, queue: asyncio.Queue):
        """Отписка; пустые наборы подписчиков удаляются"""
        subscribers = self._subscribers.get(model_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            del self._subscribers[model_id]
        logger.debug(f"Подписчик метрик модели(id:{model_id}) отключён")

    def publish(self, model_id: str):
        """Уведомление всех подписчиков модели об обновлении её метрик"""
        for queue in self._subscribers.get(model_id, ()):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                # Подписчик перечитает полный снимок по уже лежащему уведомлению
                pass

def format_sse(event: str, data: str) -> str:
    """Кадр Server-Sent Events: имя события и однострочный JSON в data"""
    return f"event: {event}\ndata: {data}\n\n"
