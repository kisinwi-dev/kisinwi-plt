from crewai import Crew
from app.core.llm import llm

from .agents import new_agent_ml_engineer
from .tasks import new_task_search_best_model

def create_crew_ml_eng(
        role: str,
        number_engineer: int|None = None,
        previous_output: str = ""
    ):
    """Создает Crew для анализа датасета"""
    
    agent = new_agent_ml_engineer(role, number_engineer)
    task = new_task_search_best_model(role, number_engineer, previous_output)

    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )


def run_engine_training_pipeline(
        role,
        number_engineer: int|None = None,
        previous_output: str = ""
):
    return create_crew_ml_eng(role, number_engineer, previous_output).kickoff()