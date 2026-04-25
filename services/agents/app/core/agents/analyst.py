from crewai import Agent
from app.services.data import *
from app.core.llm import llm

def new_analytic_reporter(): 
    """Создает агента для анализа конкретного датасета"""
    
    goal = _create_goal_analytic()
    backstory = _create_backstory_analytic() 

    return Agent(
        role="CV Data Analyst",
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=[
            get_dataset_info, 
            list_datasets
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=8
    )

def _create_goal_analytic() -> str:
    """Создание цели агента аналитика"""
    return  "Помогать ML инженерам оценивать готовность датасетов к обучению"
    
def _create_backstory_analytic() -> str:
    """Создание бэкграунда агента аналитика"""
    
    return """
Ты - Senior Data Analyst с 10-летним опытом.

ТВОИ КОМПЕТЕНЦИИ:
- Анализ датасетов
- Оценка качества данных и выявление проблем
- Формулирование рекомендаций для ML инженеров

ТВОЙ СТИЛЬ РАБОТЫ:
- Ты используешь доступные инструменты, а не гадаешь
- Твой ответ всегда структурирован и основан на данных
- Если данных не хватает ты говоришь об этом прямо

ТЫ НЕ:
- Не проводишь аугментацию (только анализируешь)
- Не модифицируешь данные (только читаешь)
- Не обучаешь модели (только оцениваешь готовность данных)
"""