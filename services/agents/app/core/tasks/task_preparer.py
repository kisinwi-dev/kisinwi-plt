from crewai import Task, Agent
from typing import Optional, List

from app.core.agents.task_preparer import new_agent_task_preparer

def new_task_task_preparer(
    previous_output: List[str],
    dataset_id: str | None = None,
    version_id: str | None = None,
    agent: Agent | None = None,
    extra: str | None = None
) -> Task:
    """
    Создать задачу для подготовки итогового JSON
    
    Args:
        previous_output: Рекомендации от ML инженеров
        dataset_id: Id датасета с которым мы работаем,
        version_id: Id версии с которой мы работаем,
        agent: Агент
        extra: Дополнительная информация для агента
    """
    
    task_description = _get_task_desc(
        previous_output=previous_output, 
        dataset_id=dataset_id,
        version_id=version_id,
        extra=extra
    )
    expected_output = _get_expected_output_template()

    return Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent if agent else new_agent_task_preparer()
    )

def _get_task_desc(
    previous_output: List[str], 
    dataset_id: str | None = None,
    version_id: str | None = None,
    extra: str | None = None
) -> str:
    """
    Получить описание задачи для подготовки json
    
    Args:
        dataset_id: Id датасета с которым мы работаем,
        version_id: Id версии с которой мы работаем,
        extra: Дополнительная информация (К примеру версия dataset, про которую не шла речь у инженеров)
    """

    context = ""

    if dataset_id is not None and version_id is not None:
        context += "Информация об используемом датасете:\n"
        context += f"dataset_id: {dataset_id}"
        context += f"version_id: {version_id}"

    if previous_output:
        context = "ИСХОДНЫЕ ДАННЫЕ (от ML-инженеров):\n"
        for output in previous_output:
            context += "\n"+output

    if extra is not None:
        context += f"Дополнительная информация: \n{extra}"

    return f"""
На основе анализов ML-инженеров сформируй итоговый JSON конфиг для сервиса обучения.

{context}

ПРАВИЛА:
1. Не выдумывай — бери значения из анализов ML-инженеров
2. Если есть противоречия — выбери вариант, который встречается чаще
3. Названия функций должны быть идентичны PyTorch
4. Выведи ТОЛЬКО JSON, без пояснений
"""

def _get_expected_output_template() -> str:
    """Шаблон ожидаемого вывода"""
    
    return """ТОЛЬКО JSON, без пояснений. Подробности можешь узнать в инструменте."""