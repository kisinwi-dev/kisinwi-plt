import time
from typing import Callable

from crewai import Crew, CrewOutput

from app.services.agent_history import agent_history_client
from app.services.metrics import add_agent_in_metrics
from app.core.memory import agent_response_context
from app.logs import get_logger

logger = get_logger(__name__)


def run_crew_with_tracking(
    crew: Crew,
    agent_role: str,
    inputs: dict,
    get_result_text: Callable[[CrewOutput], str] = lambda o: o.raw if hasattr(o, "raw") else str(o),
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

    agent_history_client.agent_start(
        response_id=response_id,
        agent_role=agent_role,
        text=f"Агент '{agent_role}' начал работу",
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
            text=get_result_text(crew_output),
            duration_ms=duration_ms,
        )
        return crew_output

    except Exception as e:
        agent_history_client.error(f"Агент '{agent_role}' завершился с ошибкой: {str(e)}")
        raise

    finally:
        agent_response_context.clear()
