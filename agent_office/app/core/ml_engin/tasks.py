from crewai import Task
from .agents import new_agent_ml_engineer

def new_task_search_best_model(
        number_engineer: int|None = None,
        previous_output: str = ""
) -> Task:
    return Task(
            description=f"""Проанализируй следующие данные о датасете и предложи архитектуру модели:

{previous_output}

На основе этих данных дай рекомендации по архитектуре.""",
        expected_output="Рекомендации по архитектуре и обучению модели",
        agent=new_agent_ml_engineer(number_engineer)
    )
