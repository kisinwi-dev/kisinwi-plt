from typing import Optional
from crewai import Task, Agent

from app.core.agents.analyst_data import new_analytic_reporter

def new_task_data_analytic(
    dataset_id: str,
    version_id: Optional[str] = None, 
    agent: Optional[Agent] = None
) -> Task:
    """Создать задачу для аналитики данных"""

    task_description = _get_task_desc(dataset_id, version_id)
    expected_output = _get_expected_output_template(dataset_id, version_id)

    return Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent if agent else new_analytic_reporter()
    )

def _get_task_desc(dataset_id: str, version_id: str | None = None) -> str:
    """Получить детальное описание задачи агента-аналитика"""

    if version_id:
        focus_text = f"версию {version_id} датасета {dataset_id}"
    else:
        focus_text = f"все версии датасета {dataset_id}"

    return f"""
Проведи анализ {focus_text}.

ШАГ 1: Получи данные через get_dataset_info('{dataset_id}')

ШАГ 2: Ответь на вопросы:
   2.1. Основные характеристики: сколько классов? какие имена? какая задача?
   2.2. Версии: {'сравни все версии и выбери лучшую' if not version_id else 'проанализируй указанную версию'}
   2.3. Баланс классов: есть ли дисбаланс? соотношение max/min?
   2.4. Качество: пустые сплиты? все ли классы присутствуют? форматы?

ШАГ 3: Выяви проблемы по шкале:
   🔴 КРИТИЧНЫЕ: обучение невозможно
   🟡 ВАЖНЫЕ: ухудшат качество  
   🟢 НЕТ ПРОБЛЕМ: можно обучать

ШАГ 4: Дай рекомендации.

ОГРАНИЧЕНИЯ:
- Только факты из get_dataset_info
- Не выдумывай цифры
- Не предлагай аугментацию
"""

def _get_expected_output_template(dataset_id: str, version_id: str | None = None) -> str:
    """Шаблон ожидаемого вывода"""
    
    return f"""
Отчет в формате:

📊 СТАТУС: [ГОТОВ / НЕ ГОТОВ / ТРЕБУЕТ ДОРАБОТКИ]

📈 ОСНОВНЫЕ ХАРАКТЕРИСТИКИ:
- Датасет: {dataset_id}
- Классов: [число]
- Имена классов: [список]
- Количество версий: [число]
{"- Анализируемая версия: " + version_id if version_id else "- Рекомендуемая версия: [какая и почему]"}

🔍 ДЕТАЛЬНЫЙ АНАЛИЗ:
- Всего изображений: [число]
- Размер: [число] MB
- Train/Val/Test: [число] / [число] / [число]
- Баланс классов: [СБАЛАНСИРОВАН / УМЕРЕННЫЙ ДИСБАЛАНС / СИЛЬНЫЙ ДИСБАЛАНС]
- Соотношение max/min: [число]:1
- Основные форматы: [формат] ([число] шт, [процент]%)
- Проблемы: [список или "нет"]

🎯 РЕКОМЕНДАЦИИ:
🔴 КРИТИЧНО: [проблема → решение или "нет"]
🟡 ВАЖНО: [рекомендация или "нет"]

✅ ГОТОВНОСТЬ К ОБУЧЕНИЮ: [ДА/НЕТ] потому что [причина]
"""