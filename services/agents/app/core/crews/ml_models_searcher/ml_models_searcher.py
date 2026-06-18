from typing import List
from pydantic import BaseModel, Field
from crewai.tools import tool

from ..utils import AgentOutput
from app.core.memory import models_context
from app.services.ml_models import ml_models_client
from app.logs import get_logger

logger = get_logger(__name__)

# Раньше это был отдельный LLM-агент. Теперь сводка собирается детерминированно из
# уже готового metrics_report версий (его пишет metrics_analyst), поэтому свой LLM
# не нужен. AGENT_ROLE оставлен строкой для трекинга/истории.
AGENT_ROLE = "Local Model Search Agent"


class MetricSummary(BaseModel):
    model_name: str = Field(..., description="Имя модели и её версия")
    summary_metric_info: str = Field(..., description="Краткая информация о метриках модели")


class MLModelsSearcherOutput(AgentOutput):
    """Формат выхода поисковика по обученным моделям"""
    text: str = Field(..., description="Подробное описание всех моделей и их метрик")
    summary: str = Field(..., description="Краткий вывод о лучшей модели и общем качестве")
    metrics_summary: List[MetricSummary] = Field(description="Сводка метрик по версиям моделей")

    def to_history_text(self) -> str:
        parts = [
            "## 🔎 Поиск обученных моделей",
            self.text,
            f"**Вывод:** {self.summary}",
        ]
        if self.metrics_summary:
            metrics = "\n".join(
                f"- **{m.model_name}:** {m.summary_metric_info}"
                for m in self.metrics_summary
            )
            parts.append(f"**Метрики по моделям:**\n{metrics}")
        return "\n\n".join(parts)


def run_ml_models_searcher(
    model_ids: List[str],
    context: str = "",
    verbose: bool = False
) -> MLModelsSearcherOutput:
    """
    Детерминированная сводка по обученным ранее версиям моделей.

    Для каждой версии берём её разбор метрик (metrics_report), который ранее
    записал metrics_analyst — повторно анализировать LLM-ом не нужно.

    Args:
        model_ids: Список ID версий моделей для сводки
        context: Контекст (не используется, оставлен для совместимости сигнатуры)
        verbose: Не используется, оставлен для совместимости сигнатуры
    """
    text_parts: List[str] = []
    metrics_summary: List[MetricSummary] = []

    for version_id in model_ids:
        version = ml_models_client.get_version(version_id)
        if version is None:
            logger.warning(f"ML Models Searcher: версия {version_id} не найдена")
            continue
        name = f"{version.get('model_type', 'модель')} v{version.get('version', '?')}"
        report = version.get("metrics_report") or "Описание метрик отсутствует."
        text_parts.append(
            f"### {name}\n"
            f"Статус: {version.get('status', '—')}\n"
            f"Анализ метрик:\n{report}"
        )
        metrics_summary.append(MetricSummary(model_name=name, summary_metric_info=report))

    if not metrics_summary:
        logger.info("ML Models Searcher: обученных ранее моделей нет")
        return MLModelsSearcherOutput(
            text="Ранее обученных моделей нет.",
            summary="",
            metrics_summary=[],
        )

    logger.info(f"ML Models Searcher завершён | Моделей разобрано: {len(metrics_summary)}")
    return MLModelsSearcherOutput(
        text="\n\n".join(text_parts),
        summary=f"Обучено ранее версий: {len(metrics_summary)}. Разбор каждой — в анализе метрик.",
        metrics_summary=metrics_summary,
    )


@tool("MLModelsSearcher")
def tool_run_ml_models_searcher(
    context: str
) -> str:
    """
    НАЗНАЧЕНИЕ: Получить информцию об обученных ранее ML моделях.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно проанализировать успешные и неудачные эксперименты
    - Для поиска лучшей модели среди уже обученных
    - Чтобы не повторять ошибки прошлых экспериментов
    - Для понимания, какие архитектуры уже пробовали

    ВХОДНЫЕ ДАННЫЕ:
    - context: Контекст поиска моделей

    ВОЗВРАЩАЕТ:
    - Структурированный ответ с информацией об обученных моделях
    """
    result = run_ml_models_searcher(
        model_ids=models_context.get_models(),
        context=context,
    )
    return result.to_history_text()
