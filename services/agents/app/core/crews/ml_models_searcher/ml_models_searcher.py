from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from app.services.ml_models import get_ml_models_info, get_all_ml_models_info
from app.services.metrics import get_metrics
from app.services.metrics.post import add_agent_in_metrics
from app.services.agent_history.post import add_reponse_in_history
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

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
            tools= [
                get_ml_models_info,
                get_all_ml_models_info,
                get_metrics
            ]
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
    Запускает агента-поисковика лучших практик.
    
    Args:
        discussion_id: ID дискуссии
        model_ids: Конкретный запрос для поиска
        context: Дополнительный контекст
        verbose: Логирование
    """
    crew = MLModelsSearcherCrew().crew(verbose=verbose)

    crew_output = crew.kickoff(
        inputs={
            "model_ids": ",".join(id for id in model_ids),
            "context": context 
        }
    )
    result: MLModelsSearcherOutput

    if not isinstance(crew_output, CrewOutput):
        return MLModelsSearcherOutput(
            text="В процессе работы была получена ошибка с типизацией",
            summary="",
            metrics_summary=[]
        )

    try:

        task_output = crew_output.tasks_output[0]
        result = task_output.pydantic # type: ignore[index]

    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        raw_text = extract_result(crew_output)
        result = MLModelsSearcherOutput(
            text=raw_text,
            summary="",
            metrics_summary=[]
        )

    # Сохраняем метрики и историю
    add_agent_in_metrics(crew=crew)

    add_reponse_in_history(
        response_id=str(crew.id),
        agent_role=crew.agents[0].role,
        agent_response=result.text  # сохраняем основной текст
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
    model_ids: str,
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
    - model_ids: ID моделей (Пример: '["model_123", "model_456"]')
    - context: Контекст поиска моделей

    ВОЗВРАЩАЕТ:
    - Структурированный ответ с информацией об обученных моделях
    """
    result = run_ml_models_searcher(
        model_ids=[model_ids],
        context=context
    )

    result_str = "# Ответ Поисковика моделей"
    result_str += f"\n## Подробное описание всех моделей:\n{result.text}"
    result_str += f"\n## Краткий вывод о лучшей модели:\n{result.summary}"
    
    for metric_summary in result.metrics_summary:
        result_str += f"\n## Модель: {metric_summary.model_name}\n{metric_summary.summary_metric_info}"

    return result_str