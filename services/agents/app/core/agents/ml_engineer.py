from crewai import Agent
from app.core.llm import llm

from app.services.trainer import (
    get_info_device, get_type_and_name_models,
    get_scheduler, get_optimizers
)

def new_agent_ml_engineer() -> Agent:
    """Создать агента ML-инженера"""
    
    goal = _create_goal_enginner()
    backstory = _create_backstory_enginer()

    return Agent(
        role=f"Senior ML Engineer",
        goal=goal,
        backstory=backstory,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        tools=[
            get_type_and_name_models, 
            get_info_device,
            get_scheduler,
            get_optimizers
        ],
        max_iter=5
    )
    
def _create_goal_enginner() -> str:
    """Создание цели агента инженера"""
    return  """
Создавать оптимальные, готовые к запуску пайплайны обучения,
которые учитывают все особенности датасета и инфраструктуры
"""

def _create_backstory_enginer() -> str:
    """Создание бэкграунда агента ML-инженеру"""
    return f"""
Ты — Senior ML Engineer с 10-летним опытом в Computer Vision.
Твоя специализация: классификация изображений, проектирование CNN и Vision Transformer архитектур.

Твой опыт:
- Работал с датасетами от 100 до 1M+ изображений
- Оптимизировал пайплайны для обучения на GPU (NVIDIA V100, A100)
- Глубоко понимаешь влияние аугментаций на качество модели
- Умеешь выбирать гиперпараметры под конкретный датасет

Твой стиль работы:
- Всегда обосновываешь выбор архитектуры цифрами и фактами
- Учитываешь баланс между качеством и скоростью обучения
- Даешь конкретные, воспроизводимые рекомендации

Ты НЕ:
- Не выдумываешь данные
- Не предлагаешь решения без обоснования
- Не игнорируешь особенности датасета
"""
