from crewai import Task, Agent
from typing import Optional, List

from app.core.agents.task_preparer import new_agent_task_preparer

def new_task_task_preparer(
    previous_output: Optional[List[str]] = None,
    agent: Optional[Agent] = None,
    extra: str | None = None
) -> Task:
    """
    Создать задачу для подготовки итогового JSON
    
    Args:
        extra: дополнительная информация для агента
    """
    
    task_description = _get_task_desc(previous_output, extra=extra)
    expected_output = _get_expected_output_template()

    return Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent if agent else new_agent_task_preparer()
    )

def _get_task_desc(
    previous_output: Optional[List[str]] = None, 
    extra: str | None = None
) -> str:
    """
    Получить описание задачи для подготовки json
    
    Args:
        extra: Дополнительная информация (К примеру версия dataset, про которую не шла речь у инженеров)
    """
    
    context = ""
    if previous_output:
        context = "ИСХОДНЫЕ ДАННЫЕ (от ML-инженеров):"
        for output in previous_output:
            context += " \n"+output
        
    
    return f"""
На основе анализов ML-инженеров сформируй итоговый JSON конфиг для сервиса обучения.

{context}

{'Дополнительная информация: '+ extra if extra else ''}

ПРАВИЛА:
1. Не выдумывай — бери значения из анализов ML-инженеров
2. Если есть противоречия — выбери вариант, который встречается чаще
3. Названия функций должны быть идентичны PyTorch
4. Выведи ТОЛЬКО JSON, без пояснений
"""

def _get_expected_output_template() -> str:
    """Шаблон ожидаемого вывода"""
    
    return """ТОЛЬКО JSON, без пояснений. Подробности можешь узнать в инструменте."""