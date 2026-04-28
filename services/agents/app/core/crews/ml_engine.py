from typing import Optional, Tuple, List
from crewai import Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.types.usage_metrics import UsageMetrics

from app.core.agents.ml_engineer import new_agent_ml_engineer
from app.core.tasks.ml_engine import new_task_search_best_model


def create_ml_ensemble_crew(
    num_engineers: int = 1,
    info_data: str = "",
    verbose: bool = True,
    agents: List[BaseAgent] | None = None,
    tasks: List[Task] | None = None
) -> Crew:
    """
    Создает Crew из нескольких ML инженеров
    
    Args:
        num_engineers: Количество инженеров,
        info_data: Описание имеющихся данных,
        verbos: Логирование в консоли,
        agents: Агенты
        tasks: Задачи для агентов

    Returns:
        Crew
    """
    
    # Если есть готвовые агенты и задачи
    if agents is not None and tasks is not None:
        return Crew(
            agents=agents, 
            tasks=tasks, 
            verbose=verbose
        )
    
    agents = []
    tasks = []
    
    for i in range(num_engineers):
        agent = new_agent_ml_engineer() 
        task = new_task_search_best_model(
            number_engineer=i+1,
            info_data=info_data,
            agent=agent
        )
        agents.append(agent)
        tasks.append(task)
    
    return Crew(agents=agents, tasks=tasks, verbose=verbose)

def run_ml_engineering(
    num_engineers: int = 1,
    info_data: str = "",
    verbose: bool = True
) -> Tuple[list[str], UsageMetrics]:
    """
    Запуск ML инженеров
    
    Args:
        num_engineering: Количество инженеров
        info_data: Информация о данных на которых будет обучаться модель
        verbose: Логирование работы агентов
    
    Returns:
        Tuple[list[str], UsageMetrics] - список результатов рассуждений агентов и 
        метрики потраченных токенов.
    """
    
    crew = create_ml_ensemble_crew(
        num_engineers=num_engineers,
        info_data=info_data,
        verbose=verbose
    )
    
    try:
        crew.kickoff()
        results = []
        for task in crew.tasks:
            if task.output and hasattr(task.output, 'raw'):
                results.append(task.output.raw)
            else:
                results.append(str(task.output))
        
        metrics = crew.usage_metrics
        
        if metrics is None:
            raise ValueError("Метрики работы агента не получены")
        
        return results, metrics
    except Exception as e:
        raise RuntimeError(f"Ошибка в работе ML-инженеров: {str(e)}")