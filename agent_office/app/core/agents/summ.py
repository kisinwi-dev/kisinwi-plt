from crewai import Agent

from app.core.llm import llm
from app.services.trainer import get_example_run_config_trainer_json, get_type_and_name_models, get_info_device


def new_agent_task_preparer():
    """Агент для подготовки итогового JSON на основе всех предыдущих анализов"""
    
    return Agent(
        role="ML Task Configuration Engineer",
        goal="На основе всех предыдущих анализов подготовить итоговый JSON для сервиса задач обучения",
        backstory=create_backstory_preparer(),
        llm=llm,
        tools=[
            get_example_run_config_trainer_json, 
            get_type_and_name_models, 
            get_info_device
        ],
        verbose=True,
        allow_delegation=False,
    )

    
def create_backstory_preparer():
    """Создание бэкграунда агенту создающему задачи"""

    return f"""Ты финальный агент в цепочке анализа.


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

Твой ответ - чистый JSON, который пойдет в сервис задач."""