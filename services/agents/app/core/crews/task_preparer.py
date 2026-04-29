from typing import Optional, Tuple, List, Union
from crewai import Crew, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.types.usage_metrics import UsageMetrics

from app.core.agents.task_preparer import new_agent_task_preparer
from app.core.tasks.task_preparer import new_task_task_preparer
from app.services.metrics.post import add_agent_in_metrics

def create_summary_crew(
    previous_outputs: List[str],
    verbose: bool = True,
    dataset_id: str | None = None,
    version_id: str | None = None,
    agent: BaseAgent | None = None,
    task: Task | None = None,
    extra: str | None = None
) -> Crew:
    """
    Создает Crew для подготовки итогового JSON
    
    Args:
        previous_outputs: Список результатов от ML инженеров
        verbose: Подробный вывод
        dataset_id: Id датасета с которым мы работаем,
        version_id: Id версии с которой мы работаем,
        agent: Агент-подготовщик (если None - создается новый)
        task: Задача (если None - создается новая)
        extra: Дополнительная информация для агента
    
    Returns:
        Crew: Сконфигурированная команда для подготовки JSON
    """

    preparer = agent if agent else new_agent_task_preparer()
    summary_task = task if task else new_task_task_preparer(
        previous_outputs, 
        dataset_id=dataset_id,
        version_id=version_id,
        extra=extra
    )
    
    return Crew(
        agents=[preparer],
        tasks=[summary_task],
        verbose=verbose
    )


def run_create_task_params_json(
    previous_outputs: List[str],
    verbose: bool = True,
    dataset_id: str | None = None,
    version_id: str | None = None,
    extra: str | None = None,
    conversation_id: str | None = None
) -> Tuple[str, UsageMetrics]:
    """
    Запуск агента для подготовки итогового JSON
    
    Args:
        previous_outputs: Список результатов от ML инженеров
        verbose: Подробный вывод
        dataset_id: Id датасета с которым мы работаем,
        version_id: Id версии с которой мы работаем,
        extra: дополнительная информация для агента
        conversation_id: id диалога
    
    Returns:
        Tuple[str], UsageMetrics]:
            - результат (JSON конфиг)
            - метрики использования токенов
    """
    
    crew = create_summary_crew(
        previous_outputs=previous_outputs,
        verbose=verbose,
        dataset_id=dataset_id,
        version_id=version_id,
        extra=extra
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
        raise Exception(f"Ошибка при подготовке итогового JSON: {str(e)}")
