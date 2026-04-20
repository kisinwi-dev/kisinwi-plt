from crewai import Agent
from app.core.llm import llm


def new_agent_ml_engineer(
        role_specific: str, 
        number_engineer: int|None = None
) -> Agent:
    """ML-инженер"""
    
    return Agent(
        role=f"Senior ML Engineer - {role_specific}",
        goal=f"На основе анализа данных подготовить технические рекомендации",
        backstory=create_backstory_enginer(role_specific, number_engineer),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    
def create_backstory_enginer(
        role_specific: str, 
        number_engineer: int|None = None
) -> str:
    """Создание бэкграунда агенту ML-инженеру"""

    name_enginer = f"ML_ENGINEER_{role_specific.upper().replace(' ', '_')}"

    if number_engineer is not None:
        name_enginer += f"_{number_engineer}"

    return f"""Ты ML-инженер, специализирующийся на {role_specific}.

ТВОЯ ЗАДАЧА:
Проанализируй эти данные и дай рекомендации.

ФОРМАТ ОТВЕТА:
=== {name_enginer}_START ===
[твои рекомендации]
=== {name_enginer}_END ===

ВСЕ МОДЕЛИ НА ВЫХОДЕ У НАС ДОЛЖНЫ БЫТЬ МУЛЬТИКЛАССОВЫМИ

Используй ТОЛЬКО данные из предыдущего анализа. Не выдумывай."""
