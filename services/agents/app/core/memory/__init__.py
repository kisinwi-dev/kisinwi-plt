from typing import Optional, List
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


class ModelsContext:
    def __init__(self):
        self._models: ContextVar[Optional[List[str]]] = ContextVar('models_context', default=None)

    def add_model(self, model_id: str) -> None:
        """Добавить ID модели"""
        models = self._models.get()
        if models is None:
            models = []
        models.append(model_id)
        self._models.set(models)
        logger.debug(f"ModelsContext: добавлен model_id={model_id}, всего: {len(models)}")

    def get_models(self) -> List[str]:
        """Получить список всех ID моделей"""
        return self._models.get() or []
    
    def clear(self) -> None:
        """Очистить контекст моделей"""
        self._models.set([])
        logger.info("ModelsContext: очищен")
    
    def get_models_info(self) -> str:
        """
        Получить строку с ID моделей для передачи агенту.
        Удобно для использования в промптах.
        """
        models = self._models.get()
        if not models:
            return "Модели ещё не создавались"
        
        return f"Созданные модели (ID): {', '.join(models)}"
    
    def __enter__(self):
        """Поддержка контекстного менеджера"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Автоматическая очистка при выходе из контекста"""
        self.clear()

class AgentResponseContext:
    def __init__(self):
        self._response_id: ContextVar[Optional[str]] = ContextVar('agent_response_id', default=None)

    def set(self, response_id: str) -> None:
        self._response_id.set(response_id)

    def get(self) -> Optional[str]:
        return self._response_id.get()

    def clear(self) -> None:
        self._response_id.set(None)

    def is_set(self) -> bool:
        return self._response_id.get() is not None


class IterationContext:
    def __init__(self):
        self._iteration: ContextVar[Optional[int]] = ContextVar('iteration', default=None)

    def set(self, iteration: int) -> None:
        self._iteration.set(iteration)

    def get(self) -> Optional[int]:
        return self._iteration.get()

    def clear(self) -> None:
        self._iteration.set(None)


class LlmModelContext:
    """
    Override LLM-модели на конкретный запуск пайплайна.

    Если установлен — имеет приоритет над глобально выбранной моделью.
    Используется в app.core.llm.resolve_model_id().
    """
    def __init__(self):
        self._model_id: ContextVar[Optional[str]] = ContextVar('llm_model', default=None)

    def set(self, model_id: str) -> None:
        self._model_id.set(model_id)
        logger.info(f"LlmModelContext: установлен llm_model={model_id}")

    def get(self) -> Optional[str]:
        return self._model_id.get()

    def clear(self) -> None:
        self._model_id.set(None)

    def is_set(self) -> bool:
        return self._model_id.get() is not None


models_context = ModelsContext()
discussion_context = Discussion()
agent_response_context = AgentResponseContext()
iteration_context = IterationContext()
llm_model_context = LlmModelContext()