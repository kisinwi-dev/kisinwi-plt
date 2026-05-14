from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from app.services.metrics.post import add_agent_in_metrics
from app.services.agent_history.post import add_reponse_in_history
from app.services.data import (
    get_dataset_info,
    get_version_info,
    list_datasets,
)
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

class DatasetAnalystOut(BaseModel):
    brief_description: str = Field(description="Краткое описание датасета")
    quality_assessment: str = Field(description="Оценка качества данных: чистота, полнота, консистентность")
    found_issues: str = Field(description="Список найденных проблем")
    recommendations: str = Field(description="Список рекомендаций по обработке данных и улучшению качества")
    readiness_assessment: bool = Field(description="Оценка готовности к обучению: 'Готов', 'Не готов'")

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
            tools=[
                get_dataset_info,
                get_version_info,
                list_datasets,
            ],
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

    * Автоматически записывает метрики использования агента, а так же
    записывает в историю дискусии.
    """
    crew = DatasetAnalystCrew().crew(verbose=verbose)

    crew_output = crew.kickoff(
        inputs={
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
        }
    )

    result: DatasetAnalystOut

    if not isinstance(crew_output, CrewOutput):
        return DatasetAnalystOut(
            brief_description="Не получилось обработать результат ответа агента.",
            quality_assessment="",
            found_issues="",
            recommendations="Попробуйте перезагрузить систему",
            readiness_assessment=False
        )

    try:

        task_output = crew_output.tasks_output[0]
        result = task_output.pydantic # type: ignore[index]

    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        raw_text = extract_result(crew_output)
        result = DatasetAnalystOut(
            brief_description=raw_text,
            quality_assessment="",
            found_issues="",
            recommendations="Не удалось обработать ответ агента в 'pydantic' схему",
            readiness_assessment=False
        )

    # Сохраняем метрики и историю
    add_agent_in_metrics(crew=crew)

    add_reponse_in_history(
        response_id=str(crew.id),
        agent_role=crew.agents[0].role,
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