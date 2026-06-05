import time
from typing import Optional

from crewai import Crew, CrewOutput

from app.services.agent_history import agent_history_client
from app.services.metrics import add_agent_in_metrics
from app.core.memory import agent_response_context, iteration_context
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


def _default_result_text(o: CrewOutput) -> str:
    """
    Текст ответа агента для истории.

    Если у задачи есть pydantic-вывод с методом to_history_text() — берём его
    человекочитаемый markdown-нарратив. Иначе откатываемся на сырой вывод LLM.
    """
    try:
        pyd = o.tasks_output[0].pydantic
        if pyd is not None and hasattr(pyd, "to_history_text"):
            return pyd.to_history_text()
    except Exception:
        pass
    return o.raw if hasattr(o, "raw") else str(o)


def run_crew_with_tracking(
    crew: Crew,
    agent_role: str,
    inputs: dict,
) -> CrewOutput | None:
    """
    Запускает crew.kickoff с полным жизненным циклом трекинга.
    response_id = str(crew.id) — согласован с метриками.

    Returns:
        CrewOutput или None если crew.kickoff вернул не CrewOutput
    """
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
            return None

        duration_ms = (time.time() - start_time) * 1000
        add_agent_in_metrics(crew=crew)
        agent_history_client.agent_succeed(
            response_id=response_id,
            agent_role=agent_role,
            text=_default_result_text(crew_output),
            duration_ms=duration_ms,
            model=model,
            task_name=task_name,
            iteration=iteration,
        )
        return crew_output

    except Exception as e:
        agent_history_client.error(f"Агент '{agent_role}' завершился с ошибкой: {str(e)}")
        raise

    finally:
        agent_response_context.clear()
