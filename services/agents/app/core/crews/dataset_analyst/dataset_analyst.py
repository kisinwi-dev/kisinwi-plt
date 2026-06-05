from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from .tools import get_tools
from ..utils import get_agent_role_from_config, run_crew_with_tracking
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "dataset_analyst",
    Path(__file__)
)

class DatasetAnalystOut(BaseModel):
    brief_description: str = Field(description="Краткое описание датасета")
    quality_assessment: str = Field(description="Оценка качества данных: чистота, полнота, консистентность")
    found_issues: str = Field(description="Список найденных проблем")
    recommendations: str = Field(description="Список рекомендаций по обработке данных и улучшению качества")
    readiness_assessment: bool = Field(description="Оценка готовности к обучению: true — готов к обучению, false — не готов")

    def get_summary(self) -> str:
        """Получить краткую сводку для передачи другим агентам"""
        summary = f"""
📊 ОТЧЕТ ПО ДАТАСЕТУ:

Краткое описание: 
{self.brief_description}

Качество: 
{self.quality_assessment}

Проблемы: 
{self.found_issues}

Рекомендации:
{self.recommendations}:

Готовность к обучению: {'✅ Готов' if self.readiness_assessment else '🟥 Не готов'}
""".strip()
        return summary
    
    def get_short_info(self) -> str:
        """Получить краткую сводку для передачи другим агентам"""
        summary = f"""
📊 ОТЧЕТ ПО ДАТАСЕТУ:

Краткое описание: 
{self.brief_description}

Качество: 
{self.quality_assessment}

Проблемы: 
{self.found_issues}
""".strip()
        return summary


@CrewBase
class DatasetAnalystCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def dataset_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["dataset_analyst"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=8,
        )

    @task
    def dataset_analyst_task(self) -> Task:
        return Task(
            config=self.tasks_config["dataset_analyst_task"], # type: ignore[index]
            output_pydantic=DatasetAnalystOut
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

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
        },
    )

    if crew_output is None:
        return DatasetAnalystOut(
            brief_description="Не получилось обработать результат ответа агента.",
            quality_assessment="",
            found_issues="",
            recommendations="Попробуйте перезагрузить систему",
            readiness_assessment=False
        )

    try:
        result = crew_output.tasks_output[0].pydantic  # type: ignore[index]
    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        result = DatasetAnalystOut(
            brief_description=extract_result(crew_output),
            quality_assessment="",
            found_issues="",
            recommendations="Не удалось обработать ответ агента в 'pydantic' схему",
            readiness_assessment=False
        )

    logger.info("Аналитик датасетов отработал")
    return result


def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)

@tool("DatasetAgent")
def tool_run_dataset_analyst(
        dataset_id: str,
        dataset_version_id: str,
        verbose: bool = False
    ) -> str:
    """
    НАЗНАЧЕНИЕ: Получить анализ датасета.
    
    КОГДА ИСПОЛЬЗОВАТЬ:
    - Перед генерацией идей для понимания данных
    - Для оценки готовности датасета к обучению
    
    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id: ID датасета
    - dataset_version_id: ID версии датасета
    
    ВОЗВРАЩАЕТ:
    - Анализ датасета в str формате
    """
    return run_dataset_analyst(
        dataset_id,
        dataset_version_id,
        verbose
    ).get_summary()