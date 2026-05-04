from crewai import Agent
from app.core.llm import llm

from app.services.trainer import (
    get_info_device, get_type_and_name_models,
    get_scheduler, get_optimizers
)

def new_agent() -> Agent:
    """Создать агента ML-инженера"""
    
    goal = _create_goal()
    backstory = _create_backstory()

    return Agent(
        role="Информатор",
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5
    )
    
def _create_goal() -> str:
    """Создание цели агента работающего с сервисом ML моделей"""
    return  """
Создавать описание для созданных ML моделей по имеющимся данным
"""

def _create_backstory() -> str:
    """Создание бэкграунда агента ML-инженеру"""
    return f""""""
