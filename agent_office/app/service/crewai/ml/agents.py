from crewai import Agent
from .tools import get_example_run_config_trainer_json, get_type_and_name_models
from app.service.crewai.llm import llm


def new_agent_ml_engineer(role_specific: str):
    """ML-инженер"""
    
    return Agent(
        role=f"Senior ML Engineer - {role_specific}",
        goal=f"На основе анализа данных подготовить технические рекомендации",
        backstory=f"""Ты ML-инженер, специализирующийся на {role_specific}.

ТВОЯ ЗАДАЧА:
Проанализируй эти данные и дай рекомендации.

ФОРМАТ ОТВЕТА:
=== ML_ENGINEER_{role_specific.upper().replace(' ', '_')}_START ===
[твои рекомендации]
=== ML_ENGINEER_{role_specific.upper().replace(' ', '_')}_END ===

ВСЕ МОДЕЛИ НА ВЫХОДЕ У НАС ДОЛЖНЫ БЫТЬ МУЛЬТИКЛАССОВЫМИ

Используй ТОЛЬКО данные из предыдущего анализа. Не выдумывай.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

def new_agent_task_preparer(previous_outputs: list):
    """Агент для подготовки итогового JSON на основе всех предыдущих анализов"""
    
    # Объединяем все предыдущие выводы
    combined_analysis = "\n\n".join(previous_outputs)
    
    return Agent(
        role="ML Task Configuration Engineer",
        goal="На основе всех предыдущих анализов подготовить итоговый JSON для сервиса задач обучения",
        backstory=f"""Ты финальный агент в цепочке анализа.

ПОЛУЧЕННЫЕ ДАННЫЕ ОТ ВСЕХ АГЕНТОВ:
{combined_analysis}

ТВОЯ ЗАДАЧА:
На основе всех предыдущих анализов (от ML-инженеров) создать единый JSON объект.

Требования к json ты получишь из инструмента get_example_run_config_trainer_json и get_type_and_name_models

ПРАВИЛА:
1. ВНИМАТЕЛЬНО изучи все предыдущие анализы
2. Объедини информацию в единый структурированный JSON
3. Используй реальные данные, не выдумывай
4. Выведи ТОЛЬКО JSON, без лишнего текста
5. JSON должен быть валидным и готовым к отправке

Название в loss_fn_config, optimizer_config, scheduler_config должны быть идентичны названием функций из PyTorch.

Твой ответ - чистый JSON, который пойдет в сервис задач.""",
        llm=llm,
        tools=[get_example_run_config_trainer_json, get_type_and_name_models],
        verbose=True,
        allow_delegation=False,
    )
