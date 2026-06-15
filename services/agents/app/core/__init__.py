"""
Пакет ядра агентов.

Намеренно лёгкий: не импортирует crews/пайплайны на уровне пакета, чтобы импорт
любого подмодуля (например, app.core.memory) не тянул весь граф crews. Иначе
низкоуровневый app.services.agent_history, зависящий от app.core.memory, замыкал
бы цикл импорта через crews → agent_history.

development_models ре-экспортируется лениво (PEP 562), публичный API
`from app.core import development_models` сохраняется.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.development import development_models

__all__ = ["development_models"]


def __getattr__(name: str):
    if name == "development_models":
        from app.core.development import development_models
        return development_models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
