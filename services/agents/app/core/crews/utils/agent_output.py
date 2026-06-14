from typing import Optional

from pydantic import BaseModel


class AgentOutput(BaseModel):
    """
    Единый контракт ответа агента.

    Помимо структурных полей (их переиспользуют другие агенты и пайплайн),
    каждый ответ умеет отдавать человекочитаемый markdown-нарратив «что сделал
    агент и почему». Этот нарратив идёт и в историю агентов, и в передачу между
    агентами — единое представление вместо разрозненных get_*-методов.
    """

    def to_history_text(self) -> str:
        """Markdown-нарратив ответа агента. Переопределяется в наследниках."""
        raise NotImplementedError


def extract_raw_text(crew_output) -> str:
    """Сырой текстовый вывод crew: raw первой задачи или строковое представление."""
    if crew_output is None:
        return ""
    if hasattr(crew_output, "raw"):
        return crew_output.raw
    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw
    return str(crew_output)


def first_task_pydantic(crew_output) -> Optional[BaseModel]:
    """Pydantic-вывод первой задачи или None (crew_output пуст / нет pydantic)."""
    if crew_output is None:
        return None
    try:
        return crew_output.tasks_output[0].pydantic
    except Exception:
        return None
