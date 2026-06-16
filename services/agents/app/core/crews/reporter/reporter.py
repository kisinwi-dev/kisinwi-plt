from pathlib import Path
from typing import List
from pydantic import Field
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools import get_tools
from ..utils import get_agent_role_from_config, run_crew_with_tracking, AgentOutput, extract_raw_text, with_modifier
from app.core.memory import models_context, discussion_context
from app.logs import get_logger
from app.core.llm import get_llm_precise

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "reporter",
    Path(__file__)
)

class ReporterOut(AgentOutput):
    result: str = Field(description="Готова ли модель к обучению")
    description: str = Field(description="Описание работы агентов и результаты обученных ими моделей")

    def to_history_text(self) -> str:
        return "\n\n".join([
            "## 📋 Итоговый отчёт",
            f"**Готовность модели:** {self.result}",
            self.description,
        ])

@CrewBase
class ReporterCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=with_modifier(self.agents_config["reporter"]),  # type: ignore[index]
            verbose=True,
            llm=get_llm_precise(),
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=5,
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
    Агент репортер подводит итоги обучений.

    Args:
        business_requirements: бизнес требования
        deployment_constraints: технические требования
        verbose: логирование в консоли
    """
    crew = ReporterCrew().crew(verbose=verbose)

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={
            "models_id": models_context.get_models(),
            "discussion_id": discussion_context.get(),
            "business_requirements": business_requirements,
            "deployment_constraints": deployment_constraints,
        },
    )

    if crew_output is None:
        return ReporterOut(
            result="Не получилось обработать результат ответа агента",
            description=""
        )

    try:
        result = crew_output.tasks_output[0].pydantic  # type: ignore[index]
    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        result = ReporterOut(
            result="Не удалось обработать ответ агента в 'pydantic' схему",
            description=extract_raw_text(crew_output)
        )

    logger.info("Reporter отработал")
    return result
