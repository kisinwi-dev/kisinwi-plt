from crewai import Task
from .agents import new_analytic_reporter

def new_task_analytic(
        dataset_id: str,   
        version_id: str|None = None
    ):
    task_description = f"Проанализируй датасет {dataset_id}"
    if version_id:
        task_description += f", версию {version_id}"

    return Task(
        description=task_description,
        expected_output="Детальный анализ датасета с рекомендациями",
        agent=new_analytic_reporter(dataset_id, version_id)
    )