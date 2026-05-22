from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from app.services.metrics.post import add_agent_in_metrics
from app.services.agent_history.post import agent_history_client
from app.services.data import get_dataset_info, list_datasets
from app.services.trainer import (
    get_example_run_config_trainer_json,
    get_type_and_name_models,
    get_info_device,
    get_scheduler,
    get_optimizers,
    get_metrics
)
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

class MlDebugerOut(BaseModel):
    """Формат ответа ML Инженера"""
    decision: bool = Field(description="Решение. Можем исправить(True) или нет(False).")
    reason: str = Field(description="Развёрнутое обоснование решения.")
    configuration: str | None = Field(
        default=None,
        description="Конфигурация для сервиса обучения"
    )

@CrewBase
class MLDebugerCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def ml_debuger(self) -> Agent:
        return Agent(
            config=self.agents_config["ml_debuger"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            tools=[
                get_example_run_config_trainer_json,
                get_type_and_name_models,
                get_info_device,
                get_scheduler,
                get_optimizers,
                get_dataset_info,
                get_metrics,
                list_datasets
            ],
            allow_delegation=False,
            max_iter=8,
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

    * Автоматически записывает метрики использования агента, а так же
    записывает в историю дискусии.
    """
    crew = MLDebugerCrew().crew(verbose=verbose)
    agent_role = crew.agents[0].role

    # Заносим в историю информацию о старте агента
    agent_history_client.agent_start(agent_role)

    crew_output = crew.kickoff(
        inputs={
            "error": error,
            "config": config
        }
    )

    result: MlDebugerOut

    if not isinstance(crew_output, CrewOutput):
        return MlDebugerOut(
            decision=False,
            reason="",
        )

    try:

        task_output = crew_output.tasks_output[0]
        result = task_output.pydantic # type: ignore[index]

    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        raw_text = extract_result(crew_output)
        result = MlDebugerOut(
            decision=False,
            reason=raw_text,
        )

    # Сохраняем метрики и историю
    add_agent_in_metrics(crew=crew)

    agent_history_client.add_response(
        response_id=str(crew.id),
        agent_role=agent_role,
        agent_response=result.reason  # сохраняем основной текст
    )

    logger.info(f"ML Engineer отработал | Задача принята в обработку: {result.decision}")
    return result

def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)
