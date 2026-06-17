import json
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from ..utils import (
    get_agent_role_from_config, run_crew_with_tracking,
    AgentOutput, with_modifier,
)
from app.config import config_base_llm, config_url
from app.logs import get_logger
from app.core.llm import get_llm_precise
from app.services.ml_models import ml_models_client
from app.services.utils import get_json

logger = get_logger(__name__)

METRICS_URL = config_url.METRICS['url']

AGENT_ROLE = get_agent_role_from_config(
    "metrics_analyst",
    Path(__file__)
)

class MetricsAnalystResponse(AgentOutput):
    """Формат ответа аналитика метрик."""
    requirements_met: bool = Field(
        description="Достигнуты ли требования пользователя. "
        "True — модель удовлетворяет требованиям, новый этап обучения не нужен. "
        "False — требования не достигнуты, нужен ещё этап обучения."
    )
    reason: str = Field(description="Обоснование вердикта по требованиям.")
    analysis: str = Field(description="Развёрнутый разбор метрик модели и её слабых мест.")

    def to_history_text(self) -> str:
        verdict = (
            "✅ Требования достигнуты — новый этап не нужен"
            if self.requirements_met
            else "🟥 Требования не достигнуты — нужен ещё этап обучения"
        )
        return "\n\n".join([
            "## 📊 Анализ метрик",
            f"**Вердикт:** {verdict}",
            f"**Обоснование:**\n{self.reason}",
            f"**Разбор метрик:**\n{self.analysis}",
        ])

@CrewBase
class MetricAnalystCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def metrics_analyst(self) -> Agent:
        return Agent(
            config=with_modifier(self.agents_config["metrics_analyst"]),  # type: ignore[index]
            verbose=True,
            llm=get_llm_precise(),
            allow_delegation=False,
            max_iter=1,
            max_execution_time=config_base_llm.AGENT_MAX_EXECUTION_TIME,
        )

    @task
    def metrics_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["metrics_analysis_task"], # type: ignore[index]
        )

    @crew
    def crew(
        self, 
        verbose: bool = False
      ) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=verbose
        )

def _fetch_metrics_data(model_id: str) -> Optional[str]:
    """
    Детерминированно собирает метрики версии + детали модели в текстовый блок
    для промпта. Раньше эти HTTP-запросы оркестрировал сам LLM через инструменты
    (лишние round-trip'ы) — теперь тащим напрямую.

    Returns:
        Текстовый блок для подстановки в промпт; None — если метрик нет
        (анализировать нечего).
    """
    exists = get_json(f"{METRICS_URL}/models/{model_id}/exists")
    if not exists.get("exists", False):
        return None

    metrics = get_json(f"{METRICS_URL}/models/{model_id}")
    details = ml_models_client.get_version(model_id) or {}

    return "\n".join([
        "## Детали версии модели",
        json.dumps({
            "version": details.get("version"),
            "model_type": details.get("model_type"),
            "status": details.get("status"),
            "train_params": details.get("train_params"),
        }, ensure_ascii=False, indent=2),
        "",
        "## Метрики (по эпохам, train/val/test)",
        json.dumps(metrics, ensure_ascii=False, indent=2),
    ])


def run_metrics_analyst(
        model_id: str,
        business_goal: str,
        verbose: bool = False
    ) -> MetricsAnalystResponse:
    """
    Агент аналитик метрик анализирует результаты работы ML модели и решает,
    достигнуты ли требования пользователя (нужен ли ещё этап обучения).

    Метрики и детали версии тащим детерминированно и отдаём агенту готовым
    текстом — это один LLM-вызов вместо серии tool-round-trip'ов. Итоговый
    разбор пишем в metrics_report версии: его потом без LLM переиспользуют
    searcher и история версий.

    Args:
        model_id: Id версии модели для анализа
        business_goal: Требования бизнеса
        verbose: Логирование в консоли
    """
    try:
        metrics_data = _fetch_metrics_data(model_id)
    except Exception as e:
        logger.warning(f"Не удалось получить метрики модели {model_id}: {e}")
        metrics_data = None

    if metrics_data is None:
        # Метрик нет/недоступны — анализировать нечего. Консервативно считаем
        # требования не достигнутыми, чтобы пайплайн не остановился впустую.
        return MetricsAnalystResponse(
            requirements_met=False,
            reason="Метрики модели недоступны — анализ невозможен.",
            analysis="",
        )

    crew = MetricAnalystCrew().crew(verbose=verbose)

    result, _raw = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={"business_goal": business_goal, "metrics_data": metrics_data},
        output_model=MetricsAnalystResponse,
    )

    if not isinstance(result, MetricsAnalystResponse):
        # LLM не вернул структуру — считаем требования не достигнутыми, чтобы
        # пайплайн не остановился на непроверенной модели.
        result = MetricsAnalystResponse(
            requirements_met=False,
            reason="Аналитик метрик не вернул структурированный вердикт.",
            analysis="",
        )

    # Сохраняем разбор как описание метрик версии (best-effort — клиент глотает
    # ошибки сети). Переиспользуется searcher'ом и историей версий без LLM.
    ml_models_client.update_version(model_id, metrics_report=result.to_history_text())

    return result
