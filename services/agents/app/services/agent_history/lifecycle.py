from contextlib import contextmanager
from typing import Iterator, Optional

from app.core.memory import discussion_context
from app.logs import get_logger
from .client import agent_history_client

logger = get_logger(__name__)


@contextmanager
def track_discussion(
    discussion_id: str,
    pipeline: str,
    title: Optional[str],
    agent_roles: list[str],
) -> Iterator[None]:
    """
    Жизненный цикл дискуссии в истории агентов.

    Создаёт дискуссию, выставляет discussion_id в контекст и гарантированно
    финализирует статус: `completed` при штатном завершении блока, `failed` при
    исключении. Ошибки самого agent_history не роняют пайплайн (клиент их глотает).
    """
    discussion_context.set(discussion_id)
    try:
        agent_history_client.create_discussion(
            discussion_id=discussion_id,
            pipeline=pipeline,
            agent_roles=agent_roles,
            title=title,
        )
        yield
        agent_history_client.update_discussion_meta(discussion_id, "completed")
    except Exception:
        agent_history_client.update_discussion_meta(discussion_id, "failed")
        raise
    finally:
        discussion_context.clear()
