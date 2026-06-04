from pathlib import Path
from typing import List
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
    "metrics_analyst",
    Path(__file__)
)

@CrewBase
class MetricAnalystCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def metrics_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["metrics_analyst"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=8,
        )

    @task
    def metrics_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["metrics_analysis_task"] # type: ignore[index]
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

def run_metrics_analyst(
        model_id: str,
        business_goal: str,
        verbose: bool = False
    ) -> str:
    """
    Агент аналитик метрик анализирует результаты работы ML модели.

    Args:
        model_id: Id модели для анализа
        business_goal: Требования бизнеса
        verbose: Логирование в консоли
    """
    crew = MetricAnalystCrew().crew(verbose=verbose)

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={"model_id": model_id, "business_goal": business_goal},
    )

    return extract_result(crew_output) if crew_output is not None else ""


def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)

@tool("DatasetAgent")
def tool_run_metrics_analyst(
        model_id: str,
        business_goal: str
    ) -> str:
    """
    НАЗНАЧЕНИЕ: Получить информацию о метриках модели.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Если требуется информация об эфективности обученной модели

    ВХОДНЫЕ ДАННЫЕ:
    - model_id: ID модели
    - business_goal: Требования к модели от бизнеса

    ВОЗВРАЩАЕТ:
    - Анализ метрик в str формате
    """
    return run_metrics_analyst(
        model_id=model_id,
        business_goal=business_goal
    )
