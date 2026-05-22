from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from app.services.metrics.post import add_agent_in_metrics
from app.services.agent_history.post import agent_history_client
from app.services.trainer import get_example_run_config_trainer_json
from app.core.crews.ml_models_searcher.ml_models_searcher import tool_run_ml_models_searcher
from app.core.crews.praxis_searcher.praxis_searcher import tool_run_praxis_searcher
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

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
            tools= [
                tool_run_praxis_searcher,
                tool_run_ml_models_searcher,
                get_example_run_config_trainer_json
            ]
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
    agent_role = crew.agents[0].role

    # Заносим в историю информацию о старте агента
    agent_history_client.agent_start(agent_role)

    denied_hypotheses_info_str = ""
    for denied_hypothesis in denied_hypotheses_info:
        denied_hypotheses_info_str += f"\nОтвергнутая гипотеза:\n{denied_hypothesis}"

    inputs = {
        "business_requirements": business_requirements,
        "dataset_info": dataset_info,
        "denied_hypotheses_info": denied_hypotheses_info_str
    }

    logger.debug('Запуск ResearcherCrew')
    crew_output = crew.kickoff(inputs=inputs)
    result: ResearcherOutput

    if not isinstance(crew_output, CrewOutput):
        return ResearcherOutput(
            analysis_summary="В процессе работы была получена ошибка с типизацией",
            hypotheses_1="",
            hypotheses_2="",
            hypotheses_3=""
        )

    try:

        task_output = crew_output.tasks_output[0]
        result = task_output.pydantic # type: ignore[index]

    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        raw_text = extract_result(crew_output)
        result = ResearcherOutput(
            analysis_summary=raw_text,
            hypotheses_1="",
            hypotheses_2="",
            hypotheses_3=""
        )

    # Сохраняем метрики и историю
    add_agent_in_metrics(crew=crew)

    agent_history_client.add_response(
        response_id=str(crew.id),
        agent_role=agent_role,
        agent_response=result.get_full_info()
    )

    logger.info(f"Researcher завершён")
    return result

def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)
