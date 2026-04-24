from crewai import Task
from .agents import new_agent_task_preparer

def new_task_summary(
        previous_output: list = []
    ):
    return Task(
        description=f"Подготовка JSON. Нельзя придумывать и менять требования. Рассуждения: {previous_output}",
        expected_output="ТОЛЬКО JSON НА ВЫХОДЕ, БЕЗ ТЕКСТА, ТОЛЬКО JSON",
        agent=new_agent_task_preparer()
    )