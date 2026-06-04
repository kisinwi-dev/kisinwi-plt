from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool

from ..utils import get_agent_role_from_config, run_crew_with_tracking
from app.logs import get_logger
from app.core.llm import llm
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
    relevance_score: Optional[int] = Field(None, ge=1, le=10, description="Оценка релевантности от 1 до 10")

class PraxisSearchOutput(BaseModel):
    """Стандартизированный вывод Praxis Searcher"""
    text: str = Field(..., description="Основной текст с найденными лучшими практиками и рекомендациями")
    sources: List[SearchSource] = Field(default_factory=list, description="Список источников с ссылками")
    summary: str = Field(..., description="Краткий summary ключевых практик")

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
            config=self.agents_config["praxis_searcher"],  # type: ignore[index]
            verbose=True,
            llm=llm,
            allow_delegation=False,
            max_iter=15,
            tools=get_tools(AGENT_ROLE)
        )

    @task
    def praxis_search_task(self) -> Task:
        return Task(
            config=self.tasks_config["praxis_search_task"],  # type: ignore[index]
            output_pydantic=PraxisSearchOutput
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

    crew_output = run_crew_with_tracking(
        crew=crew,
        agent_role=AGENT_ROLE,
        inputs={"search_query": search_query, "context": context},
    )

    if crew_output is None:
        return PraxisSearchOutput(
            text="В процессе работы была получена ошибка с типизацией",
            sources=[],
            summary="ошибка"
        )

    try:
        result = crew_output.tasks_output[0].pydantic  # type: ignore[index]
    except Exception as e:
        logger.warning(f"Не удалось получить pydantic output: {e}. Используем fallback.")
        result = PraxisSearchOutput(
            text=extract_result(crew_output),
            sources=[],
            summary="Не удалось структурировать вывод. Используйте сырой текст выше."
        )

    return result

def extract_result(crew_output):
    if hasattr(crew_output, "raw"):
        return crew_output.raw

    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw

    return str(crew_output)

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

    result_str = "# 🌐 Результаты поиска лучших практик"
    result_str += f"\n## Запрос: {search_query}"
    result_str += f"\n## Найденные практики\n{result.text}"
    result_str += f"\n## Краткий обзор\n{result.summary}"

    return result_str
