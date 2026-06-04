from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from .tools import get_tools
from ..utils import get_agent_role_from_config, run_crew_with_tracking
from app.core.memory import models_context
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "ml_models_searcher",
    Path(__file__)
)

class MetricSummary(BaseModel):
    model_name: str = Field(..., description="Имя модели и её версия")
    summary_metric_info: str = Field(..., description="Краткая информация о метриках модели")

class MLModelsSearcherOutput(BaseModel):
    """Формат выхода агента"""
    text: str = Field(..., description="Подробное описание всех моделей и их метрик")
    summary: str = Field(..., description="Краткий вывод о лучшей модели и общем качестве")
    metrics_summary: List[MetricSummary] = Field(description="Сводка метрик где ключ это версия модели, а значение это описание")

@CrewBase
class MLModelsSearcherCrew:
    """Crew для поиска лучших ML практик"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def ml_models_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config["ml_models_searcher"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            max_iter=2,
            tools=get_tools(AGENT_ROLE)
        )

    @task
    def ml_models_searcher_task(self) -> Task:
        return Task(
            config=self.tasks_config["ml_models_searcher_task"],  # type: ignore[index]
            output_pydantic=MLModelsSearcherOutput
        )

    @crew
    def crew(self, verbose: bool = False) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=verbose
        )

def run_ml_models_searcher(
    model_ids: List[str],
    context: str,
    verbose: bool = False
) -> MLModelsSearcherOutput:
    """
    Запускает агента-поисковика по обученным моделям.

    Args:
        model_ids: Список ID моделей для анализа
        context: Дополнительный контекст
        verbose: Логирование
    """
    crew = MLModelsSearcherCrew().crew(verbose=verbose)

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={
            "model_ids": ",".join(model_ids),
            "context": context,
        },
    )

    if crew_output is None:
        return MLModelsSearcherOutput(
            text="В процессе работы была получена ошибка с типизацией",
            summary="",
            metrics_summary=[]
        )

    try:
        result = crew_output.tasks_output[0].pydantic  # type: ignore[index]
    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        result = MLModelsSearcherOutput(
            text=extract_result(crew_output),
            summary="",
            metrics_summary=[]
        )

    logger.info(f"ML Models Searcher завершён | Моделей разобрано: {len(model_ids)}")
    return result

def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)

@tool("MLModelsSearcher")
def tool_run_ml_models_searcher(
    context: str
) -> str:
    """
    НАЗНАЧЕНИЕ: Получить информцию об обученных ранее ML моделях.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно проанализировать успешные и неудачные эксперименты
    - Для поиска лучшей модели среди уже обученных
    - Чтобы не повторять ошибки прошлых экспериментов
    - Для понимания, какие архитектуры уже пробовали

    ВХОДНЫЕ ДАННЫЕ:
    - context: Контекст поиска моделей

    ВОЗВРАЩАЕТ:
    - Структурированный ответ с информацией об обученных моделях
    """
    result = run_ml_models_searcher(
        model_ids=models_context.get_models(),
        context=context
    )

    result_str = "# Ответ Поисковика моделей"
    result_str += f"\n## Подробное описание всех моделей:\n{result.text}"
    result_str += f"\n## Краткий вывод о лучшей модели:\n{result.summary}"
    
    for metric_summary in result.metrics_summary:
        result_str += f"\n## Модель: {metric_summary.model_name}\n{metric_summary.summary_metric_info}"

    return result_str