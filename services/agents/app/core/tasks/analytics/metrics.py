from crewai import Task, Agent

from app.core.agents.analysts.metrics import new_analytic as new_agent_metrics_analytic

def new_analysis(
    task_id: str,
    dataset_id: str, 
    version_id: str,
    bus_req: str | None = None,
    agent: Agent | None = None
) -> Task:
    """Создать задачу для аналитики метрик"""

    task_description = _get_task_desc(task_id, dataset_id, version_id, bus_req)
    expected_output = _get_expected_output_template(task_id)

    return Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent if agent else new_agent_metrics_analytic()
    )

def _get_task_desc(
    task_id: str,
    dataset_id: str, 
    version_id: str,
    bus_req: str | None = None
) -> str:
    """Получить детальное описание задачи агента-аналитика метрик"""

    if bus_req:
        bus_req = f"""
БИЗНЕС ТРЕБОВАНИЯ К МОДЕЛИ:
{bus_req}  
"""
    else: 
        bus_req = ''

    return f"""
Проведи анализ метрик качества ML модели для задачи {task_id}.

## ШАГ 1: Получение данных

Вызови инструмент `get_metrics` с параметром task_id = '{task_id}' для получения метрик
Вызови инструмент `get_version_split_info` с параметрами dataset_id = '{dataset_id}', version_id = '{version_id}'
для получения информации о распределении данных в датасете


## ШАГ 2: Анализ метрик
Оцени каждую метрику по шкале:
- ✅ ЦЕЛЬ ДОСТИГНУТА: значение >= порога
- ⚠️ ТРЕБУЕТ ВНИМАНИЯ: значение близко к порогу (отклонение < 10%)
- ❌ КРИТИЧЕСКОЕ: значение ниже порога

Проверь:
2.1. **Основные метрики**: precision, recall, f1, accuracy
2.2. **Дисбаланс классов**: macro/micro/weighted avg
2.3. **ROC-AUC** (для бинарной классификации)
2.4. **Временные тренды**: ухудшается ли качество?

## ШАГ 3: Диагностика проблем
Определи тип проблем:

🔴 **КРИТИЧЕСКИЕ** (обучение бессмысленно):
- Метрики на валидации ≈ random (0.5 для бинарной классификации)
- Нет положительных предсказаний (recall = 0)
- Переобучение: train >> val (разница > 20%)

🟡 **СЕРЬЁЗНЫЕ** (модель плохая):
- F1 < 0.6
- Сильный дисбаланс метрик (precision 0.9, recall 0.3)
- Метрики упали на 10% за последние итерации

🟢 **НЕЗНАЧИТЕЛЬНЫЕ** (можно улучшить):
- Метрики стабильны, но ниже целевых на < 10%
- Небольшой дисбаланс классов

## ОГРАНИЧЕНИЯ:
- Только анализ метрик, без их модификации
- Если метрики не найдены, явно указать "ДАННЫХ НЕТ"
- Не додумывать значения

{bus_req}
"""

def _get_expected_output_template(task_id: str) -> str:
    """Шаблон ожидаемого вывода аналитика метрик"""
    
    return f"""
# 📊 Отчет по метрикам модели: `{task_id}`

## 🎯 Общая оценка
**Статус модели**: [ГОТОВА К PRODUCTION / ТРЕБУЕТ ДОРАБОТКИ / НЕ ПРИГОДНА]

---

## 📈 Метрики качества

| Метрика | Значение | Целевое | Статус |
|---------|----------|---------|--------|
| Accuracy | [value] | [target] | ✅/⚠️/❌ |
| Precision | [value] | [target] | ✅/⚠️/❌ |
| Recall | [value] | [target] | ✅/⚠️/❌ |
| F1-score | [value] | [target] | ✅/⚠️/❌ |
| ROC-AUC | [value] | [target] | ✅/⚠️/❌ |

**Детали по классам** (если есть дисбаланс):
- Class 0: precision/recall/f1 = [values]
- Class 1: precision/recall/f1 = [values]

---

## 🔍 Диагностика проблем

### 🔴 КРИТИЧЕСКИЕ
[список или "Отсутствуют"]

### 🟡 СЕРЬЁЗНЫЕ  
[список или "Отсутствуют"]

### 🟢 НЕЗНАЧИТЕЛЬНЫЕ
[список или "Отсутствуют"]

---

## 📉 Тренды
- F1-score: [растёт/падает/стабилен] на [X]% за [N] итераций
- Precision-Recall баланс: [ухудшился/улучшился/не изменился]

---

## ✅ Итоговое заключение
**Направление развития**: [Что стоит учесть ML-инженерам для улучшения модели]
**Ключевые метрики, требующие внимания**: [список]

"""