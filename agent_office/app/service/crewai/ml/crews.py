# app/service/crewai/analytic/crews.py
from crewai import Crew
from app.service.crewai.llm import llm

from .agents import new_agent_ml_engineer, new_agent_task_preparer
from .tasks import new_task_search_best_model, new_task_summary

def create_crew_ml_eng(
        role: str = "img classification",
        previous_output: str = ""
    ):
    """Создает Crew для анализа датасета"""
    
    agent = new_agent_ml_engineer(role)
    task = new_task_search_best_model(role, previous_output)

    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

def create_crew_summary(
        history: list = []
    ):
    """Создает Crew для анализа датасета"""
    
    agent = new_agent_task_preparer(history)
    task = new_task_summary(history)

    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )



def run_search_params_json(
        role: str = "img classification",
        previous_output: str = ""
):
    result = []

    crew = create_crew_ml_eng(role, previous_output)
    ml_result = crew.kickoff().raw
    result.append(ml_result)

    result_json = create_crew_summary(result)
    return result_json.kickoff()