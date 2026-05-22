from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from app.core.memory import models_context, discussion_context
from app.services.metrics.post import add_agent_in_metrics
from app.services.agent_history import get_agent_history
from app.services.agent_history.post import agent_history_client
from app.services.metrics import get_metrics
from app.services.ml_models import get_ml_models_info
# from app.services.da
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

class ReporterOut(BaseModel):
    result: str = Field(description="Готова ли модель к обучению")
    description: str = Field(description="Описание работы агентов и результаты обученных ими моделей")

    def get_summary(self) -> str:
        """Получить краткую сводку для передачи другим агентам"""
        summary = f"Результат обучения: {self.result}\n"
        summary += f"\n{self.description}"
        return summary

@CrewBase
class ReporterCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["reporter"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            tools=[
                get_agent_history,
                get_metrics,
                get_ml_models_info
            ],
            allow_delegation=False,
            max_iter=8,
        )

    @task
    def reporter_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporter_task"], # type: ignore[index]
            output_pydantic=ReporterOut
        )

    @crew
    def crew(
        self, 
        verbose: bool = False
      ) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=verbose
        )

def run_reporter(
        business_requirements: str,
        deployment_constraints: str,
        verbose: bool = False
    ) -> ReporterOut:
    """
    Агент аналитик данных анализирует данные. И на выдаёт описание данных.

    Args:
        business_requirements: бизнес требования
        deployment_constraints: технические требования
        verbose: логирование в консоли

    * Автоматически записывает метрики использования агента, а так же
    записывает в историю дискусии.
    """
    crew = ReporterCrew().crew(verbose=verbose)
    agent_role = crew.agents[0].role

    # Заносим в историю информацию о старте агента
    agent_history_client.agent_start(agent_role)

    crew_output = crew.kickoff(
        inputs={
            "models_id": models_context.get_models(),
            "discussion_id": discussion_context.get(),
            "business_requirements": business_requirements,
            "deployment_constraints": deployment_constraints,
        }
    )

    result: ReporterOut

    if not isinstance(crew_output, CrewOutput):
        return ReporterOut(
            result="Не получилось обработать результат ответа агента",
            description=""
        )

    try:

        task_output = crew_output.tasks_output[0]
        result = task_output.pydantic # type: ignore[index]

    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        raw_text = extract_result(crew_output)
        result = ReporterOut(
            result="Не удалось обработать ответ агента в 'pydantic' схему",
            description=f"{raw_text}"
        )

    # Сохраняем метрики и историю
    add_agent_in_metrics(crew=crew)

    agent_history_client.add_response(
        response_id=str(crew.id),
        agent_role=agent_role,
        agent_response=result.get_summary()  # сохраняем основной текст
    )

    logger.info(f"Аналитик датасетов отработал")
    return result


def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)
