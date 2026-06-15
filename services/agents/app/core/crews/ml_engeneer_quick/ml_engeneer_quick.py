from pathlib import Path
from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools import get_tools
from ..utils import (
    get_agent_role_from_config, run_crew_with_tracking,
    extract_raw_text, first_task_pydantic, with_modifier,
)
from ..ml_engeneer import MlEngineerResponse
from app.services.ml_models import NO_MODEL_HISTORY
from app.logs import get_logger
from app.core.llm import get_llm_precise

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "ml_engineer_quick",
    Path(__file__)
)


@CrewBase
class MLEngineerQuickCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def ml_engineer_quick(self) -> Agent:
        return Agent(
            config=with_modifier(self.agents_config["ml_engineer_quick"]),  # type: ignore[index]
            verbose=True,
            llm=get_llm_precise(),
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=8,
        )

    @task
    def ml_engineer_quick_task(self) -> Task:
        return Task(
            config=self.tasks_config["ml_engineer_quick_task"],  # type: ignore[index]
            output_pydantic=MlEngineerResponse
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


def run_quick_ml_engineering(
        dataset_info: str,
        business_requirements: str,
        deployment_constraints: str,
        dataset_id: str,
        dataset_version_id: str,
        model_history: str = NO_MODEL_HISTORY,
        verbose: bool = False
    ) -> MlEngineerResponse:
    """
    ML инженер быстрого прогона.

    Без участия Researcher и без возможности отказаться по качеству: конфиг
    подбирается самостоятельно, decision = false допустим только при физическом
    отсутствии датасета/версии.

    Args:
        dataset_info: Информация о датасете (сырые метаданные)
        business_requirements: Требования бизнеса
        deployment_constraints: Технические требования прода
        dataset_id: ID датасета
        dataset_version_id: ID версии датасета
        model_history: История версий существующей модели (при дообучении)
        verbose: логирование в консоли
    """
    crew = MLEngineerQuickCrew().crew(verbose=verbose)

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={
            "dataset_info": dataset_info,
            "business_requirements": business_requirements,
            "deployment_constraints": deployment_constraints,
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
            "model_history": model_history,
        },
    )

    result = first_task_pydantic(crew_output)
    if result is None:
        result = MlEngineerResponse(
            decision=False,
            reason=extract_raw_text(crew_output),
            recommendations="Ошибка при обработке ответа ML Enginner (быстрый прогон), предупреди пользователя"
        )

    logger.info(f"ML Engineer (быстрый прогон) отработал | Задача принята в обработку: {result.decision}")
    return result
