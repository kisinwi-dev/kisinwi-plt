from typing import Optional, Tuple
from crewai import Crew, Agent, Task, CrewOutput
from crewai.types.usage_metrics import UsageMetrics

from app.core.agents.analyst_data import new_analytic_reporter
from app.core.tasks.analytics import new_task_data_analytic

def create_analytics_crew(
    dataset_id: str, 
    version_id: Optional[str] = None,
    agent: Optional[Agent] = None,
    task: Optional[Task] = None,
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
    
    analyst = agent if agent else new_analytic_reporter()
    analysis_task = task if task else new_task_data_analytic(dataset_id, version_id)

    return Crew(
        agents=[analyst],
        tasks=[analysis_task],
        verbose=verbose,
    )


def run_analysis(
    dataset_id: str, 
    version_id: Optional[str] = None,
    verbose: bool = True
) -> Tuple[str, UsageMetrics]:
    """
    Запуск анализа данных
    
    Args:
        dataset_id: ID датасета
        version_id: ID версии (опционально)
        verbose: Подробный вывод
    
    Returns:
        Tuple[str, UsageMetrics]: 
            - результат анализа
            - метрики использования токенов
    """
    
    crew = create_analytics_crew(dataset_id, version_id, verbose=verbose)
    
    try:
        crew_output = crew.kickoff()
        metrics = crew.usage_metrics
        
        if isinstance(crew_output, CrewOutput) and isinstance(metrics, UsageMetrics):
            return crew_output.raw, metrics
        else: 
            raise TypeError(f"Неверный тип данных: CrewOutput={type(crew_output)}, UsageMetrics={type(metrics)}")    
    except Exception as e:
        raise Exception(f"Ошибка при анализе датасета {dataset_id}: {str(e)}")