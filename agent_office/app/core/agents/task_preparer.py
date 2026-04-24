from crewai import Agent

from app.core.llm import llm
from app.services.trainer import get_example_run_config_trainer_json, get_type_and_name_models, get_info_device

def new_agent_task_preparer():
    """Создать агента для подготовки итогового JSON на основе всех предыдущих анализов"""
    
    goal = _create_goal_task_preparer()
    backstory = _create_backstory_task_preparer()

    return Agent(
        role="ML Task Configuration Engineer",
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=[
            get_example_run_config_trainer_json, 
            get_type_and_name_models, 
            get_info_device
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

def _create_goal_task_preparer() -> str:
    """Создание задачи агента подготовки задачи"""
    return  """
Сформировать валидный JSON конфигурации для сервиса обучения,
объединив все предыдущие анализы ML-инженеров в единый исполняемый пайплайн.
"""
    
def _create_backstory_task_preparer():
    """Создание бэкграунда агенту создающему задачи"""
    return f"""
Ты финальный агент в цепочке анализа.

Для выполнения задач ты используешь инструменты.

ТВОЙ ОПЫТ:
- Отлично знаешь структуру JSON конфигов для PyTorch тренировок. Ты знаешь что, все тренировки выолняются с использованием PyTorch.
- Понимаешь, как должны называться функции в loss_fn_config, optimizer_config, scheduler_config
- Умеешь выбирать лучший вариант, если ML-инженеры предложили разные решения
- Знаешь, что инструмент *get_type_and_name_models* использует библиотеку timm
- Знаешь, какие параметры обязательны, а какие опциональны

ТВОЙ СТИЛЬ РАБОТЫ:
- Внимательно анализируешь ВСЕ предыдущие ответы ML-инженеров
- Выбираешь оптимальные параметры (если есть противоречия)
- Всегда проверяешь валидность JSON перед выводом
- Никогда не выдумываешь значения

Твой ответ - чистый JSON, который пойдет в сервис задач.
"""