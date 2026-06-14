from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from .tools import get_tools
from ..utils import (
    get_agent_role_from_config, run_crew_with_tracking, AgentOutput,
    extract_raw_text, first_task_pydantic,
)
from app.services.ml_models import NO_MODEL_HISTORY
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "ml_engineer",
    Path(__file__)
)

class MlModel(BaseModel):
    description_model: str = Field(description="Описание модели")
    type: str = Field(description="Тип модели (принадлежность к архитектуреы)")
    configuration: str = Field(
        description="Конфигурация для сервиса обучения"
    )

class MlEngineerResponse(AgentOutput):
    """Формат ответа ML Инженера"""
    decision: bool = Field(description="Решение. Обучать(True) или нет(False).")
    reason: str = Field(description="Развёрнутое обоснование решения.")
    ml_model: MlModel | None = Field(default=None, description="Информация о разработываемой модели (только если decision==True)")
    recommendations: str = Field(description="Рекомендации по улучшению или альтернативные варианты")

    def to_history_text(self) -> str:
        decision = "✅ Запускаем обучение" if self.decision else "🟥 Отказ от обучения"
        parts = [
            "## 🛠️ Решение ML-инженера",
            f"**Решение:** {decision}",
            f"**Обоснование:**\n{self.reason}",
        ]
        if self.ml_model is not None:
            parts.append(f"**Модель:** {self.ml_model.type} — {self.ml_model.description_model}")
            parts.append(f"**Конфигурация обучения:**\n```\n{self.ml_model.configuration}\n```")
        if self.recommendations:
            parts.append(f"**Рекомендации:**\n{self.recommendations}")
        return "\n\n".join(parts)

@CrewBase
class MLEngineerCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def ml_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["ml_engineer"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            tools=get_tools(AGENT_ROLE),
            allow_delegation=False,
            max_iter=8,
        )

    @task
    def ml_engineer_task(self) -> Task:
        return Task(
            config=self.tasks_config["ml_engineer_task"], # type: ignore[index]
            output_pydantic=MlEngineerResponse
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

def run_ml_engineering(
        dataset_info: str,
        business_requirements: str,
        deployment_constraints: str,
        researcher_proposals: str,
        dataset_id: str,
        dataset_version_id: str,
        model_history: str = NO_MODEL_HISTORY,
        verbose: bool = False
    ) -> MlEngineerResponse:
    """
    Агент ML инженер (полный прогон)

    Args:
        dataset_info: Информация о датасете
        business_requirements: Требования бизнеса
        deployment_constraints: Технические требования
        researcher_proposals: Предложение от ресерчера
        dataset_id: ID датасета
        dataset_version_id: ID версии датасета
        model_history: История версий существующей модели (при продолжении обучения)
        verbose: логирование в консоли
    """
    crew = MLEngineerCrew().crew(verbose=verbose)

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={
            "dataset_info": dataset_info,
            "business_requirements": business_requirements,
            "deployment_constraints": deployment_constraints,
            "researcher_proposals": researcher_proposals,
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
            "model_history": model_history,
        },
    )

    result = first_task_pydantic(crew_output)
    if result is None:
        result = MlEngineerResponse(
            decision=False,
            reason=extract_raw_text(crew_output),
            recommendations="Ошибка при обработке ответа ML Enginner, предупреди пользователя"
        )

    logger.info(f"ML Engineer отработал | Задача принята в обработку: {result.decision}")
    return result
