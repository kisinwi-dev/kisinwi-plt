from typing import Tuple
from crewai import Crew, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.types.usage_metrics import UsageMetrics

from app.core.agents.ml_models_worker import new_agent
from app.core.tasks.work_with_ml_models import new_task
from app.services.metrics.post import add_agent_in_metrics
from app.services.ml_models import ModelMeta

def create_summary_crew(
    params_train: str,
    data_summary: str,
    agent: BaseAgent | None = None,
    task: Task | None = None,
    extra: str | None = None,
    verbose: bool = True
) -> Crew:
    """
    Создает Crew для подготовки итогового JSON
    
    Args:
        params_train: Параметры обучения
        data_summary: Информация о данных
        agent: Агент
        task: Задача
        extra: Дополнительная информация для агента
    
    Returns:
        Crew: Сконфигурированная команда
    """

    agent = agent if agent else new_agent()
    task = task if task else new_task(
        params_train=params_train, 
        data_summary=data_summary,
        extra=extra
    )
    
    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=verbose
    )


def run_create_desc_model(
    params_train: str,
    data_summary: str,
    extra: str | None = None,
    verbose: bool = True
) -> Tuple[ModelMeta, UsageMetrics]:
    """
    Запуск агента для подготовки итогового JSON
    
    Args:
        params_train: Параметры обучения
        data_summary: Информация о данных
        extra: Дополнительная информация для агента
        verbose: Логировать работу агента
    
    Returns:
        Tuple[ModelMeta, UsageMetrics]:
            - результат
            - метрики использования токенов
    """
    
    crew = create_summary_crew(
        params_train=params_train, 
        data_summary=data_summary,
        extra=extra,
        verbose=verbose
    )
    
    try:
        crew_output = crew.kickoff()
        metrics = crew.usage_metrics
        
        add_agent_in_metrics(
            crew=crew
        )

        if isinstance(crew_output, CrewOutput) and isinstance(metrics, UsageMetrics):
            task_output = crew_output.tasks_output[0]
            result = task_output.pydantic

            if result is None:
                raise ValueError("Агент не вернул валидный Pydantic результат")

            if not isinstance(result, ModelMeta):
                raise TypeError(f"Ожидался ModelMeta, получили: {type(result)}")

            return result, metrics
        else: 
            raise TypeError(f"Неверный тип данных: CrewOutput={type(crew_output)}, UsageMetrics={type(metrics)}")  
    except Exception as e:
        raise Exception(f"Ошибка при создании описания модели: {str(e)}")
