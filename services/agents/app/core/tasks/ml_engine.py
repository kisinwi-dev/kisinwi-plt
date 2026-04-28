from typing import Optional
from crewai import Task, Agent

from app.core.agents.ml_engineer import new_agent_ml_engineer

# __WARNING__ ПЕРЕДЕЛАТЬ ЭТОТ ФАЙЛ, ЧТО ТО НЕ ТАК С ОТСУПАМИ

def new_task_search_best_model(
    number_engineer: int | None = None,
    info_data: str = "",
    agent: Optional[Agent] = None
) -> Task:
    """
    Создать задачу для МL-инженера

    Args: 
      number_engineer: Количество инжеров
      info_data: Описание имеющихся данных
      agent: Агент

    Returns:
      Task
    """

    description = _get_task_desc(number_engineer, info_data)
    output = _get_expected_output_template(number_engineer)

    return Task(
        description=description,
        expected_output=output,
        agent=agent if agent else new_agent_ml_engineer()
    )

def _get_task_desc(
    number_engineer: Optional[int] = None,
    info_data: str = ""
) -> str:
    """Получить детальное описание задачи для ML инженера"""
    
    engineer_label = f" (инженер #{number_engineer})" if number_engineer else ""
    
    return f"""
Спроектируй пайплайн обучения на основе анализа датасета{engineer_label}.

ДАННЫЕ О ДАТАСЕТЕ:

{info_data}

ЧТО НУЖНО ОПРЕДЕЛИТЬ:

1. АРХИТЕКТУРА МОДЕЛИ:
   - Какую модель выбрать?
   - Использовать предобученные веса?
   - Почему этот выбор оптимален?

2. ГИПЕРПАРАМЕТРЫ:
   - batch_size (учитывая размер датасета)
   - epochs
   - learning_rate
   - optimizer (AdamW / SGD / Adam)
   - scheduler (ReduceLROnPlateau / CosineAnnealingLR)

3. АУГМЕНТАЦИИ:
   - Какие аугментации применить?
   - С какой вероятностью?

4. ПРОГНОЗ ОБУЧЕНИЯ:
   - Ожидаемое время одной эпохи (в минутах)
   - Итоговое время обучения (в часах)
   - Ожидаемая точность (грубая оценка)

ОГРАНИЧЕНИЯ:
- Учитывай размер датасета
- Учитывай баланс классов
- Используй ТОЛЬКО данные из анализа
- Все обоснования должны быть в поле reasoning
"""

def _get_expected_output_template(number_engineer: Optional[int] = None) -> str:
    """Шаблон ожидаемого вывода для ML инженера"""
    
    engineer_label = f"_{number_engineer}" if number_engineer else "_1"
    
    return f"""
YAML конфигурация в формате:

=== ML_ENGINEER{engineer_label}_START ===

```yaml
model:
  name: "Название модели"
  pretrained: true/false
  reasoning: "Почему выбрал эту модель"

training:
  batch_size: число
  epochs: число
  learning_rate: число
  optimizer: "Название"
  scheduler: "Название"
  reasoning: "Обоснование гиперпараметров"

augmentations:
  train:
    - name: "Аугментация1"
      probability: 0.5
    - name: "Аугментация2"
      probability: 0.3
  reasoning: "Почему эти аугментации"

estimated_metrics:
  per_epoch_minutes: число
  total_hours: число
  expected_accuracy: число (0-1)
```

=== ML_ENGINEER{engineer_label}_START ===
"""
