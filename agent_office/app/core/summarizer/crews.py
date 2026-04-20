from crewai import Crew
from app.core.llm import llm

from .agents import new_agent_task_preparer
from .tasks import new_task_summary


def create_crew_summary(
        previous_outputs: list = []
    ):
    """Создает Crew для анализа датасета"""
    
    agent = new_agent_task_preparer(previous_outputs)
    task = new_task_summary(previous_outputs)

    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

def run_create_task_params_json(
        previous_outputs: list = []
):
    return create_crew_summary(previous_outputs).kickoff()