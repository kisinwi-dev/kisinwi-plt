from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from crewai import Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from ..utils import get_agent_role_from_config, run_crew_with_tracking, AgentOutput, with_modifier
from app.config import config_base_llm
from app.logs import get_logger
from app.core.llm import get_llm_precise
from .tools import get_tools

logger = get_logger(__name__)

AGENT_ROLE = get_agent_role_from_config(
    "praxis_searcher",
    Path(__file__)
)

class SearchSource(BaseModel):
    url: str = Field(..., description="Ссылка на источник")
    title: Optional[str] = Field(None, description="Заголовок статьи / репозитория")
    short_description: str = Field(..., description="Краткое описание, почему этот источник полезен")
    # Диапазон описан в description, но НЕ через ge/le: Pydantic превратил бы их в
    # minimum/maximum в JSON Schema, а Anthropic-провайдеры (через OpenRouter) такую
    # схему integer-поля отвергают (400). Корректность подстраховываем валидатором.
    relevance_score: Optional[int] = Field(None, description="Оценка релевантности от 1 до 10")

    @field_validator("relevance_score")
    @classmethod
    def _clamp_relevance(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return max(1, min(10, v))

class PraxisSearchOutput(AgentOutput):
    """Стандартизированный вывод Praxis Searcher"""
    text: str = Field(..., description="Основной текст с найденными лучшими практиками и рекомендациями")
    sources: List[SearchSource] = Field(default_factory=list, description="Список источников с ссылками")
    summary: str = Field(..., description="Краткий summary ключевых практик")

    def to_history_text(self) -> str:
        parts = [
            "## 🌐 Поиск лучших практик",
            self.text,
            f"**Краткий обзор:** {self.summary}",
        ]
        if self.sources:
            links = "\n".join(
                f"- [{s.title or s.url}]({s.url}) — {s.short_description}"
                for s in self.sources
            )
            parts.append(f"**Источники:**\n{links}")
        return "\n\n".join(parts)

@CrewBase
class PraxisSearcherCrew:
    """Crew для поиска лучших ML практик"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent.yaml"
    tasks_config = "config/task.yaml"

    @agent
    def praxis_searcher(self) -> Agent:
        return Agent(
            config=with_modifier(self.agents_config["praxis_searcher"]),  # type: ignore[index]
            verbose=True,
            llm=get_llm_precise(),
            allow_delegation=False,
            max_iter=5,
            max_execution_time=config_base_llm.AGENT_MAX_EXECUTION_TIME,
            tools=get_tools(AGENT_ROLE)
        )

    @task
    def praxis_search_task(self) -> Task:
        return Task(
            config=self.tasks_config["praxis_search_task"],  # type: ignore[index]
        )

    @crew
    def crew(self, verbose: bool = False) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=verbose
        )

def run_praxis_searcher(
    search_query: str,
    context: str = "",
    verbose: bool = False
) -> PraxisSearchOutput:
    """
    Запускает агента-поисковика лучших практик.

    Args:
        search_query: Конкретный запрос для поиска
        context: Дополнительный контекст
        verbose: Логирование
    """
    crew = PraxisSearcherCrew().crew(verbose=verbose)

    result, raw = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={"search_query": search_query, "context": context},
        output_model=PraxisSearchOutput,
    )

    if result is None:
        result = PraxisSearchOutput(
            text=raw or "В процессе работы была получена ошибка с типизацией",
            sources=[],
            summary="Не удалось структурировать вывод. Используйте сырой текст выше."
        )

    return result

@tool("PraxisSearcher")
def tool_run_praxis_searcher(
    search_query: str,
    context: str = ""
) -> str:
    """
    НАЗНАЧЕНИЕ: Найти лучшие практики ML в интернете.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать о современных подходах к решению задачи
    - Когда нужно подтвердить гипотезу индустриальным опытом
    - Для изучения best practices по конкретной проблеме

    ВХОДНЫЕ ДАННЫЕ:
    - search_query (str): Поисковый запрос на английском
    - context (str): Дополнительный контекст (опционально)

    ВОЗВРАЩАЕТ:
    - Структурированный ответ с лучшими практиками, ссылками и рекомендациями
    """
    result = run_praxis_searcher(
        search_query=search_query,
        context=context
    )

    return f"**Запрос:** {search_query}\n\n{result.to_history_text()}"
