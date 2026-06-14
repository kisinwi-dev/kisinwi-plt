from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from .tools import get_tools
from ..utils import get_agent_role_from_config, run_crew_with_tracking, AgentOutput, extract_raw_text, with_modifier
from app.core.memory import models_context
from app.logs import get_logger
from app.core.llm import get_llm_precise

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "ml_models_searcher",
    Path(__file__)
)

class MetricSummary(BaseModel):
    model_name: str = Field(..., description="Имя модели и её версия")
    summary_metric_info: str = Field(..., description="Краткая информация о метриках модели")

class MLModelsSearcherOutput(AgentOutput):
    """Формат выхода агента"""
    text: str = Field(..., description="Подробное описание всех моделей и их метрик")
    summary: str = Field(..., description="Краткий вывод о лучшей модели и общем качестве")
    metrics_summary: List[MetricSummary] = Field(description="Сводка метрик где ключ это версия модели, а значение это описание")

    def to_history_text(self) -> str:
        parts = [
            "## 🔎 Поиск обученных моделей",
            self.text,
            f"**Вывод:** {self.summary}",
        ]
        if self.metrics_summary:
            metrics = "\n".join(
                f"- **{m.model_name}:** {m.summary_metric_info}"
                for m in self.metrics_summary
            )
            parts.append(f"**Метрики по моделям:**\n{metrics}")
        return "\n\n".join(parts)

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
            config=with_modifier(self.agents_config["ml_models_searcher"]),  # type: ignore[index]
            verbose=True,
            llm=get_llm_precise(),
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
            text=extract_raw_text(crew_output),
            summary="",
            metrics_summary=[]
        )

    logger.info(f"ML Models Searcher завершён | Моделей разобрано: {len(model_ids)}")
    return result

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

    return result.to_history_text()