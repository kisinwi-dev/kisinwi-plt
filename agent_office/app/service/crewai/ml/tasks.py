from crewai import Task
from .agents import new_agent_ml_engineer, new_agent_task_preparer

def new_task_search_best_model(
        role_specific: str,
        previous_output: str = ""
    ):
    return Task(
            description=f"""Проанализируй следующие данные о датасете и предложи архитектуру модели:

{previous_output}

На основе этих данных дай рекомендации по архитектуре.""",
        expected_output="Рекомендации по архитектуре и обучению модели",
        agent=new_agent_ml_engineer(role_specific)
    )

def new_task_summary(
        previous_output: list = []
    ):
    return Task(
        description=f"Подготовка JSON",
        expected_output="ТОЛЬКО JSON НА ВЫХОДЕ, БЕЗ ТЕКСТА, ТОЛЬКО JSON",
        agent=new_agent_task_preparer(previous_output)
    )