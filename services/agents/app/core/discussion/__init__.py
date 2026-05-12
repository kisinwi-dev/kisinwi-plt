from typing import Optional
from contextvars import ContextVar
from app.logs import get_logger

logger = get_logger(__name__)


class Discussion:
    def __init__(self):
        self._discussion_id: ContextVar[Optional[str]] = ContextVar('discussion_id', default=None)
    
    def set(self, discussion_id: str) -> None:
        """Установить discussion_id для текущего контекста"""
        self._discussion_id.set(discussion_id)
        logger.info(f"DiscussionContext: установлен discussion_id={discussion_id}")
    
    def get(self) -> str:
        """Получить discussion_id из текущего контекста"""
        disc_id = self._discussion_id.get()
        if disc_id is None:
            raise ValueError("discussion_id не установлен")
        return disc_id
    
    def clear(self) -> None:
        """Очистить discussion_id в текущем контексте"""
        self._discussion_id.set(None)
        logger.info("DiscussionContext: очищен")
    
    def is_set(self) -> bool:
        """Проверить, установлен ли discussion_id"""
        return self._discussion_id.get() is not None
    
    def __enter__(self):
        """Поддержка контекстного менеджера"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Автоматическая очистка при выходе из контекста"""
        self.clear()


discussion_context = Discussion()