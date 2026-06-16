from pathlib import Path
from typing import List
from pydantic import Field
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools import get_tools
from ..utils import (
    get_agent_role_from_config, run_crew_with_tracking, AgentOutput,
    extract_raw_text, first_task_pydantic, with_modifier,
)
from app.logs import get_logger
from app.core.llm import get_llm_precise

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "ml_debuger",
    Path(__file__)
)

class MlDebugerOut(AgentOutput):
    """Формат ответа ML Инженера"""
    decision: bool = Field(description="Решение. Можем исправить(True) или нет(False).")
    reason: str = Field(description="Развёрнутое обоснование решения.")
    configuration: str | None = Field(
        default=None,
        description="Конфигурация для сервиса обучения"
    )

    def to_history_text(self) -> str:
        decision = "✅ Ошибка исправлена" if self.decision else "🟥 Исправить не удалось"
        parts = [
            "## 🐞 Отладка обучения",
            f"**Решение:** {decision}",
            f"**Обоснование:**\n{self.reason}",
        ]
        if self.configuration:
            parts.append(f"**Исправленная конфигурация:**\n```\n{self.configuration}\n```")
        return "\n\n".join(parts)

@CrewBase
class MLDebugerCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def ml_debuger(self) -> Agent:
        return Agent(
            config=with_modifier(self.agents_config["ml_debuger"]),  # type: ignore[index]
            verbose=True,
            llm=get_llm_precise(),
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=5,
        )

    @task
    def ml_debuger_task(self) -> Task:
        return Task(
            config=self.tasks_config["ml_debuger_task"], # type: ignore[index]
            output_pydantic=MlDebugerOut
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

def run_ml_debug(
        error: str,
        config: str,
        verbose: bool = False
    ) -> MlDebugerOut:
    """
    Агент ML дебагер ищет решение проблемы возникшей в сервисе тренировок

    Args:
        error: Ошибка полученная от сервиса обучения
        config: Конфиг отправленный в сервис обучения
        verbose: логирование в консоли
    """
    crew = MLDebugerCrew().crew(verbose=verbose)

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={"error": error, "config": config},
    )

    result = first_task_pydantic(crew_output)
    if result is None:
        result = MlDebugerOut(decision=False, reason=extract_raw_text(crew_output))

    logger.info(f"ML Debuger отработал | Можем исправить: {result.decision}")
    return result
