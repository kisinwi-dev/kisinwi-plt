from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, CrewOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from app.services.metrics.post import add_agent_in_metrics
from app.services.agent_history.post import add_reponse_in_history
from app.services.trainer import (
    get_example_run_config_trainer_json,
    get_type_and_name_models,
    get_info_device,
    get_scheduler,
    get_optimizers
)
from app.logs import get_logger
from app.core.llm import llm

logger = get_logger(__name__)

class MlEngineerResponse(BaseModel):
    """Формат ответа ML Инженера"""
    decision: bool = Field(description="Решение. Обучать(True) или нет(False).")
    reason: str = Field(description="Развёрнутое обоснование решения.")
    configuration: Optional[str] = Field(
        default=None,
        description="Конфигурация для сервиса обучения (только если decision==True)"
    )
    recommendations: str = Field(description="Рекомендации по улучшению или альтернативные варианты")

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
            tools=[
                get_example_run_config_trainer_json,
                get_type_and_name_models,
                get_info_device,
                get_scheduler,
                get_optimizers
            ],
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
        business_goal: str,
        technical_goal: str,
        researcher_info: str,
        training_goal: str,
        verbose: bool = False
    ) -> MlEngineerResponse:
    """
    Агент ML инженер

    Args:
        business_goal: Требования бизнеса
        technical_goal: Технические требования
        training_goal: Задача решаемая при обучении
        researcher_info: Предложение от ресерчера
        verbose: логирование в консоли

    * Автоматически записывает метрики использования агента, а так же
    записывает в историю дискусии.
    """
    crew = MLEngineerCrew().crew(verbose=verbose)

    crew_output = crew.kickoff(
        inputs={
            "business_goal": business_goal,
            "technical_goal": technical_goal,
            "training_goal": training_goal,
            "researcher_info": researcher_info
        }
    )

    result: MlEngineerResponse

    if not isinstance(crew_output, CrewOutput):
        return MlEngineerResponse(
            decision=False,
            reason="",
            recommendations="Ошибка в ML Engineer, предупреди пользователя"
        )

    try:

        task_output = crew_output.tasks_output[0]
        result = task_output.pydantic # type: ignore[index]

    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        raw_text = extract_result(crew_output)
        result = MlEngineerResponse(
            decision=False,
            reason=raw_text,
            recommendations="Ошибка при обработке ответа ML Enginner, предупреди пользователя"
        )

    # Сохраняем метрики и историю
    add_agent_in_metrics(crew=crew)

    add_reponse_in_history(
        response_id=str(crew.id),
        agent_role=crew.agents[0].role,
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