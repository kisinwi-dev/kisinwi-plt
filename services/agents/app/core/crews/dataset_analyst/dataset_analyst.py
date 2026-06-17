from pathlib import Path
from typing import List
from pydantic import Field
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools import get_tools
from ..utils import get_agent_role_from_config, run_crew_with_tracking, AgentOutput, with_modifier
from app.config import config_base_llm
from app.logs import get_logger
from app.core.llm import get_llm_precise

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "dataset_analyst",
    Path(__file__)
)

class DatasetAnalystOut(AgentOutput):
    brief_description: str = Field(description="Краткое описание датасета")
    quality_assessment: str = Field(description="Оценка качества данных: чистота, полнота, консистентность")
    found_issues: List[str] = Field(default_factory=list, description="Найденные проблемы, по одной на элемент списка")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации по обработке данных, по одной на элемент списка")
    readiness_assessment: bool = Field(description="Оценка готовности к обучению: true — готов к обучению, false — не готов")

    def to_history_text(self) -> str:
        readiness = "✅ Готов к обучению" if self.readiness_assessment else "🟥 Не готов к обучению"

        def bullets(items: List[str]) -> str:
            return "\n".join(f"- {item}" for item in items) if items else "—"

        return "\n\n".join([
            "## 🧪 Анализ датасета",
            f"**Готовность:** {readiness}",
            f"**Что в датасете:**\n{self.brief_description}",
            f"**Качество данных:**\n{self.quality_assessment}",
            f"**Найденные проблемы:**\n\n{bullets(self.found_issues)}",
            f"**Рекомендации:**\n\n{bullets(self.recommendations)}",
        ])


@CrewBase
class DatasetAnalystCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def dataset_analyst(self) -> Agent:
        return Agent(
            config=with_modifier(self.agents_config["dataset_analyst"]),  # type: ignore[index]
            verbose=True,
            llm=get_llm_precise(),
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=5,
            max_execution_time=config_base_llm.AGENT_MAX_EXECUTION_TIME,
        )

    @task
    def dataset_analyst_task(self) -> Task:
        return Task(
            config=self.tasks_config["dataset_analyst_task"], # type: ignore[index]
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

def run_dataset_analyst(
        dataset_id: str,
        dataset_version_id: str,
        verbose: bool = False
    ) -> DatasetAnalystOut:
    """
    Агент аналитик данных анализирует данные. И на выдаёт описание данных.

    Args:
        dataset_id: id датасета
        dataset_version_id: id версии датасета
        verbose: логирование в консоли
    """
    crew = DatasetAnalystCrew().crew(verbose=verbose)

    result, raw = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
        },
        output_model=DatasetAnalystOut,
    )

    if result is None:
        result = DatasetAnalystOut(
            brief_description=raw or "Не получилось обработать результат ответа агента.",
            quality_assessment="",
            found_issues=[],
            recommendations=["Не удалось обработать ответ агента в 'pydantic' схему"],
            readiness_assessment=False
        )

    logger.info("Аналитик датасетов отработал")
    return result