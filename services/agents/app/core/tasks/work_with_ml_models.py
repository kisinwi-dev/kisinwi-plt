from crewai import Task, Agent

from app.core.agents.ml_models_worker import new_agent
from app.services.ml_models import ModelMeta

def new_task(
    params_train: str,
    data_summary: str,
    agent: Agent | None = None,
    extra: str | None = None
) -> Task:
    """
    Создать задачу для описания модели JSON
    
    Args:
        params_train: Параметры обучения
        data_summary: Информация о данных
        agent: Агент
        extra: Дополнительная информация для агента
    """
    
    task_description = _get_task_desc(
        params_train=params_train, 
        data_summary=data_summary,
        extra=extra
    )
    expected_output = _get_expected_output_template()

    return Task(
        description=task_description,
        expected_output=expected_output,
        output_pydantic=ModelMeta,
        agent=agent if agent else new_agent()
    )

def _get_task_desc(
    params_train: str, 
    data_summary: str,
    extra: str | None = None
) -> str:
    """
    Получить описание задачи для подготовки json
    
    Args:
        params_train: Параметры обучения
        data_summary: Информация о данных
        extra: Дополнительная информация (К примеру версия dataset, про которую не шла речь у инженеров)
    """

    context = f"""
Информация о датасете:
{data_summary}

Параметры обучения:
{params_train}
"""

    if extra is not None:
        context += f"Дополнительная информация: \n{extra}"

    return context

def _get_expected_output_template() -> str:
    return """
Ответ должен содержать краткое и понятное описания для человека.
"""