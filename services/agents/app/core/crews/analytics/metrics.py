from typing import Tuple
from crewai import Crew, Agent, Task, CrewOutput
from crewai.types.usage_metrics import UsageMetrics

from app.core.agents.analysts.metrics import new_analytic as new_agent_metrics_analytic
from app.core.tasks.analytics.metrics import new_analysis as new_task_metrics_analysis
from app.services.metrics.post import add_agent_in_metrics

def create_data_analysis(
    task_id: str,
    dataset_id: str, 
    version_id: str,
    bus_req: str | None = None,
    agent: Agent | None = None,
    task: Task | None = None,
    verbose: bool = True
) -> Crew:
    """
    Создает Crew для анализа датасета
    
    Args:
        task_id: ID задачи
        bus_req: Бизнес требования к модели
        agent: Агент-аналитик (если None - создается новый)
        task: Задача для анализа (если None - создается новая)
        verbose: Подробный вывод
    
    Returns:
        Crew: Сконфигурированная команда для анализа
    """
    
    analyst = agent if agent else new_agent_metrics_analytic()
    analysis_task = task if task else new_task_metrics_analysis(
        task_id, 
        dataset_id, 
        version_id,
        bus_req
    )

    return Crew(
        agents=[analyst],
        tasks=[analysis_task],
        verbose=verbose,
    )

def run_analysis(
    task_id: str, 
    dataset_id: str, 
    version_id: str,
    bus_req: str | None,
    verbose: bool = True,
    conversation_id: str | None = None
) -> Tuple[str, UsageMetrics]:
    """
    Запуск анализа данных
    
    Args:
        task_id: ID задачи
        bus_req: Бизнес требования к модели
        verbose: Подробный вывод
        conversation_id: id диалога
    
    Returns:
        Tuple[str, UsageMetrics]: 
            - результат анализа
            - метрики использования токенов
    """
    
    crew = create_data_analysis(
        task_id, 
        dataset_id=dataset_id,
        version_id=version_id,
        bus_req=bus_req,
        verbose=verbose
    )
    
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
        raise Exception(f"Ошибка при анализе метрик задачи {task_id}: {str(e)}")