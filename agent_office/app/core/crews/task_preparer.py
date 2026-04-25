from typing import Optional, Tuple, List, Union
from crewai import Crew, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.types.usage_metrics import UsageMetrics

from app.core.agents.task_preparer import new_agent_task_preparer
from app.core.tasks.task_preparer import new_task_task_preparer


def create_summary_crew(
    previous_outputs: Optional[List[str]] = None,
    verbose: bool = True,
    agent: Optional[BaseAgent] = None,
    task: Optional[Task] = None
) -> Crew:
    """
    Создает Crew для подготовки итогового JSON
    
    Args:
        previous_outputs: Список результатов от ML инженеров
        verbose: Подробный вывод
        agent: Агент-подготовщик (если None - создается новый)
        task: Задача (если None - создается новая)
    
    Returns:
        Crew: Сконфигурированная команда для подготовки JSON
    """
    
    preparer = agent if agent else new_agent_task_preparer()
    summary_task = task if task else new_task_task_preparer(previous_outputs or [])
    
    return Crew(
        agents=[preparer],
        tasks=[summary_task],
        verbose=verbose
    )


def run_create_task_params_json(
    previous_outputs: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[str, UsageMetrics]:
    """
    Запуск агента для подготовки итогового JSON
    
    Args:
        previous_outputs: Список результатов от ML инженеров
        verbose: Подробный вывод
    
    Returns:
        Tuple[str], UsageMetrics]:
            - результат (JSON конфиг)
            - метрики использования токенов
    """
    
    crew = create_summary_crew(
        previous_outputs=previous_outputs,
        verbose=verbose
    )
    
    try:
        crew_output = crew.kickoff()
        metrics = crew.usage_metrics
        
        if isinstance(crew_output, CrewOutput) and isinstance(metrics, UsageMetrics):
            return crew_output.raw, metrics
        else: 
            raise TypeError(f"Неверный тип данных: CrewOutput={type(crew_output)}, UsageMetrics={type(metrics)}")  
    except Exception as e:
        raise Exception(f"Ошибка при подготовке итогового JSON: {str(e)}")
