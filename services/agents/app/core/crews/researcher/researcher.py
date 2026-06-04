from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools import get_tools
from ..utils import get_agent_role_from_config, run_crew_with_tracking
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "researcher",
    Path(__file__)
)

class ResearcherOutput(BaseModel):
    analysis_summary: str = Field(..., description="Краткий анализ текущей ситуации")
    hypotheses_1: str = Field(..., description="Гипотеза 1 с описанием, обоснованием и ожидаемым приростом")
    hypotheses_2: str = Field(..., description="Гипотеза 2 с описанием, обоснованием и ожидаемым приростом")
    hypotheses_3: str = Field(..., description="Гипотеза 3 с описанием, обоснованием и ожидаемым приростом")

    def get_full_info(self) -> str:
        text = "📊 Краткий анализ текущей ситуации"
        text = f"\n{self.analysis_summary}"
        text += f"\n🔬 Гипотеза 1:\n{self.hypotheses_1}"
        text += f"\n🔬 Гипотеза 2:\n{self.hypotheses_2}"
        text += f"\n🔬 Гипотеза 3:\n{self.hypotheses_3}"
        return text


@CrewBase
class ResearcherCrew:
    """Crew для поиска лучших ML практик"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            allow_delegation=False,
            max_iter=15,
            tools= get_tools(AGENT_ROLE)
        )

    @task
    def researcher_task(self) -> Task:
        return Task(
            config=self.tasks_config["researcher_task"],  # type: ignore[index]
            output_pydantic=ResearcherOutput
        )

    @crew
    def crew(self, verbose: bool = False) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=verbose
        )

def run_researcher(
    business_requirements: str,
    dataset_info: str,
    denied_hypotheses_info: List[str],
    verbose: bool = False
) -> ResearcherOutput:
    """
    Запускает агента-поисковика лучших практик.

    Args:
        business_requirements: Требования бизнеса
        dataset_info: Информация о датасете
        denied_hypotheses_info: Список гипотез, отстранённых ранее
    """
    crew = ResearcherCrew().crew(verbose=verbose)

    denied_hypotheses_info_str = ""
    for denied_hypothesis in denied_hypotheses_info:
        denied_hypotheses_info_str += f"\nОтвергнутая гипотеза:\n{denied_hypothesis}"

    logger.debug('Запуск ResearcherCrew')
    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={
            "business_requirements": business_requirements,
            "dataset_info": dataset_info,
            "denied_hypotheses_info": denied_hypotheses_info_str,
        },
    )

    if crew_output is None:
        return ResearcherOutput(
            analysis_summary="В процессе работы была получена ошибка с типизацией",
            hypotheses_1="",
            hypotheses_2="",
            hypotheses_3=""
        )

    try:
        result = crew_output.tasks_output[0].pydantic  # type: ignore[index]
    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        result = ResearcherOutput(
            analysis_summary=extract_result(crew_output),
            hypotheses_1="",
            hypotheses_2="",
            hypotheses_3=""
        )

    logger.info("Researcher завершён")
    return result

def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)
