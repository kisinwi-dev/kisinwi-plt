from typing import Tuple
from crewai import Crew, Agent, Task, CrewOutput
from crewai.types.usage_metrics import UsageMetrics

from app.core.agents.analysts.datasets import new_analytic as new_agent_data_analytic
from app.core.tasks.analytics.datasets import new_analysis as new_task_data_analysis
from app.services.metrics.post import add_agent_in_metrics

def create_data_analysis(
    dataset_id: str, 
    version_id: str | None = None,
    agent: Agent | None = None,
    task: Task | None = None,
    verbose: bool = True
) -> Crew:
    """
    Создает Crew для анализа датасета
    
    Args:
        dataset_id: ID датасета
        version_id: ID версии (опционально)
        agent: Агент-аналитик (если None - создается новый)
        task: Задача для анализа (если None - создается новая)
        verbose: Подробный вывод
    
    Returns:
        Crew: Сконфигурированная команда для анализа
    """
    
    analyst = agent if agent else new_agent_data_analytic()
    analysis_task = task if task else new_task_data_analysis(dataset_id, version_id)

    return Crew(
        agents=[analyst],
        tasks=[analysis_task],
        verbose=verbose,
    )

def run_analysis(
    dataset_id: str, 
    version_id: str | None = None,
    verbose: bool = True,
    conversation_id: str | None = None
) -> Tuple[str, UsageMetrics]:
    """
    Запуск анализа данных
    
    Args:
        dataset_id: ID датасета
        version_id: ID версии (опционально)
        verbose: Подробный вывод
        conversation_id: id диалога
    
    Returns:
        Tuple[str, UsageMetrics]: 
            - результат анализа
            - метрики использования токенов
    """
    
    crew = create_data_analysis(dataset_id, version_id, verbose=verbose)
    
    try:
        crew_output = crew.kickoff()
        metrics = crew.usage_metrics
        
        add_agent_in_metrics(
            crew=crew,
            conversation_id=conversation_id if conversation_id else "no_dialog"
        )

        if isinstance(crew_output, CrewOutput) and isinstance(metrics, UsageMetrics):
            return crew_output.raw, metrics
        else: 
            raise TypeError(f"Неверный тип данных: CrewOutput={type(crew_output)}, UsageMetrics={type(metrics)}")    
    except Exception as e:
        raise Exception(f"Ошибка при анализе датасета {dataset_id}: {str(e)}")