import time
from typing import Optional, Type, Tuple

from crewai import Crew, CrewOutput
from pydantic import BaseModel

from app.services.agent_history import agent_history_client
from app.services.metrics import add_agent_in_metrics
from app.core.memory import agent_response_context, iteration_context, discussion_context
from app.core.crews.utils.agent_output import (
    extract_raw_text, parse_agent_output, output_format_hint,
)
from app.logs import get_logger

logger = get_logger(__name__)


def _extract_crew_meta(crew: Crew) -> tuple[Optional[str], Optional[str]]:
    """Возвращает (model_name, task_name). Никогда не бросает исключений."""
    model = None
    task_name = None
    try:
        if crew.agents and crew.agents[0].llm:
            model = getattr(crew.agents[0].llm, "model", None)
    except Exception:
        pass
    try:
        if crew.tasks:
            task_name = crew.tasks[0].name
    except Exception:
        pass
    return model, task_name


def _history_text(parsed: Optional[BaseModel], raw: str) -> str:
    """
    Текст ответа агента для истории.

    Если raw распарсился в модель с to_history_text() — берём её
    человекочитаемый markdown-нарратив. Иначе откатываемся на сырой вывод LLM.
    """
    if parsed is not None and hasattr(parsed, "to_history_text"):
        try:
            return parsed.to_history_text()
        except Exception:
            pass
    return raw


def run_crew_with_tracking(
    crew: Crew,
    agent_role: str,
    inputs: dict,
    output_model: Type[BaseModel],
) -> Tuple[Optional[BaseModel], str]:
    """
    Запускает crew.kickoff с полным жизненным циклом трекинга.
    response_id = str(crew.id) — согласован с метриками.

    Задачи отдают raw (без output_pydantic) — парсим его сами, устойчиво к
    «грязному» финалу агента (см. parse_agent_output). В inputs автоматически
    подмешивается output_format — подсказка схемы для промпта.

    Returns:
        (parsed_or_none, raw_text). parsed=None если kickoff не дал CrewOutput
        либо ни один JSON-объект из raw не валиден — caller делает свой fallback.
    """
    inputs.setdefault("output_format", output_format_hint(output_model))

    response_id = str(crew.id)
    agent_response_context.set(response_id)
    start_time = time.time()

    model, task_name = _extract_crew_meta(crew)
    iteration = iteration_context.get()

    agent_history_client.agent_start(
        response_id=response_id,
        agent_role=agent_role,
        text=f"Агент '{agent_role}' начал работу",
        model=model,
        task_name=task_name,
        iteration=iteration,
    )

    try:
        crew_output = crew.kickoff(inputs=inputs)

        if not isinstance(crew_output, CrewOutput):
            return None, ""

        raw = extract_raw_text(crew_output)
        parsed = parse_agent_output(raw, output_model)

        duration_ms = (time.time() - start_time) * 1000
        discussion_id = discussion_context.get() if discussion_context.is_set() else None
        add_agent_in_metrics(crew=crew, discussion_id=discussion_id)
        agent_history_client.agent_succeed(
            response_id=response_id,
            agent_role=agent_role,
            text=_history_text(parsed, raw),
            duration_ms=duration_ms,
            model=model,
            task_name=task_name,
            iteration=iteration,
        )
        return parsed, raw

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        agent_history_client.agent_error(
            response_id=response_id,
            agent_role=agent_role,
            text=f"Агент '{agent_role}' завершился с ошибкой: {str(e)}",
            duration_ms=duration_ms,
            model=model,
            task_name=task_name,
            iteration=iteration,
        )
        raise

    finally:
        agent_response_context.clear()
