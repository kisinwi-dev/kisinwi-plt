from pathlib import Path
from typing import List
from pydantic import Field
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools import get_tools
from ..utils import (
    get_agent_role_from_config, run_crew_with_tracking,
    AgentOutput, first_task_pydantic, with_modifier,
)
from app.logs import get_logger
from app.core.llm import get_llm_precise

logger = get_logger(__name__)

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
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=5,
        )

    @task
    def metrics_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["metrics_analysis_task"], # type: ignore[index]
            output_pydantic=MetricsAnalystResponse,
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

def run_metrics_analyst(
        model_id: str,
        business_goal: str,
        verbose: bool = False
    ) -> MetricsAnalystResponse:
    """
    Агент аналитик метрик анализирует результаты работы ML модели и решает,
    достигнуты ли требования пользователя (нужен ли ещё этап обучения).

    Args:
        model_id: Id модели для анализа
        business_goal: Требования бизнеса
        verbose: Логирование в консоли
    """
    crew = MetricAnalystCrew().crew(verbose=verbose)

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={"model_id": model_id, "business_goal": business_goal},
    )

    result = first_task_pydantic(crew_output)
    if isinstance(result, MetricsAnalystResponse):
        return result
    # LLM не вернул структуру — считаем требования не достигнутыми, чтобы пайплайн
    # не остановился на непроверенной модели.
    return MetricsAnalystResponse(
        requirements_met=False,
        reason="Аналитик метрик не вернул структурированный вердикт.",
        analysis="",
    )
