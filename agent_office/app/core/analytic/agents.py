from crewai import Agent
from app.services.data import *
from app.core.llm import llm

def new_analytic_reporter(
        dataset_id: str, 
        version_id: str|None = None
    ): 
    """
    Создает агента для анализа конкретного датасета
    
    Args:
        dataset_id: ID датасета для анализа
        version_id: ID версии (если None - анализирует все версии)
    """
    
    goal = create_goal_analytic(dataset_id, version_id)
    backstory = create_backstory_analytic(dataset_id, version_id) 

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
        max_iter=8,
    )

def create_goal_analytic(dataset_id: str, version_id: str|None = None):
    """Создание задачи агенту аналитику"""
    goal = f"Провести полный анализ датасета {dataset_id}"
    if version_id:
        goal += f" (версия {version_id})"
    goal += " и подготовить детальный отчёт для ML-инженеров"

    return goal
    
def create_backstory_analytic(dataset_id: str, version_id: str|None = None):
    """Создание бэкграунда агенту аналитику"""
    return f"""Ты — аналитик данных в команде Deep Learning.

ТЕКУЩАЯ ЗАДАЧА:
Анализ датасета: {dataset_id}
{f'Целевая версия: {version_id}' if version_id else 'Анализ всех версий'}

ТВОЙ ИНСТРУМЕНТ:
- get_dataset_info(dataset_id) - получить всю информацию

ТВОЙ АЛГОРИТМ:
1. Вызови get_dataset_info('{dataset_id}')
2. Проанализируй полученные данные
3. {'Сфокусируйся на версии ' + version_id if version_id else 'Проанализируй все версии'}
4. Подготовь отчёт

ЧТО ПРОАНАЛИЗИРОВАТЬ:
- Основные характеристики датасета
- {'Только указанную версию' if version_id else 'Все доступные версии и их сравнение'}
- Классы и их распределение
- Количество сэмплов
- Качество и пригодность

СТРУКТУРА ОТЧЁТА:
1. Краткая сводка
2. Детальная информация
3. Анализ {'версии' if version_id else 'версий'}
4. Выводы и рекомендации

ВАЖНО: Работай ТОЛЬКО с датасетом {dataset_id}. Не трогай другие датасеты."""
