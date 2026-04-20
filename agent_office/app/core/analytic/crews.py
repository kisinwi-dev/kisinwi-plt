from crewai import Crew
from app.core.llm import llm

from .agents import new_analytic_reporter
from .tasks import new_task_analytic

def create_crew(dataset_id: str, version_id: str|None = None):
    """Создает Crew для анализа датасета"""
    
    analyst = new_analytic_reporter(dataset_id, version_id)
    task = new_task_analytic(dataset_id, version_id)

    return Crew(
        agents=[analyst],
        tasks=[task],
        verbose=True
    )

def run_analysis(dataset_id: str, version_id: str|None = None):
    """Запуск анализа данных"""
    crew = create_crew(dataset_id, version_id)
    return crew.kickoff()