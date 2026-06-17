import os
import string
from typing import List
from crewai.tools import BaseTool
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    ArxivPaperTool
)

from ..utils import get_tools_with_tracking

_SCRAPE_MAX_CHARS = 16000  # ~4k токенов — лимит endpoint'а 400k, защита от мегабайт в контексте


def _sanitize_scrape_output(text: str) -> str:
    """Не пускать в контекст бинарь/PDF и не класть мегабайты текста.

    ScrapeWebsiteTool тянет любые ссылки, включая arXiv PDF, и возвращает сырой
    бинарь → context length exceeded. Режем такие выхлопы в источнике.
    """
    if not isinstance(text, str):
        return text
    head = text.lstrip()[:1000]
    non_printable = sum(c not in string.printable for c in head)
    if head.startswith("%PDF") or (head and non_printable / len(head) > 0.3):
        return "[skipped: binary/PDF content. Use ArxivPaperTool for papers instead.]"
    if len(text) > _SCRAPE_MAX_CHARS:
        return text[:_SCRAPE_MAX_CHARS] + "\n[...truncated]"
    return text


def _wrap_scrape(tool: BaseTool) -> BaseTool:
    original_run = tool._run

    def wrapped_run(*args, **kwargs):
        return _sanitize_scrape_output(original_run(*args, **kwargs))

    tool._run = wrapped_run
    return tool


_tool_instances: List[BaseTool] = [
    _wrap_scrape(ScrapeWebsiteTool()),
    ArxivPaperTool(),
]
# ponytail: Serper подключаем только при наличии ключа — иначе SerperDevTool падает с KeyError
if os.getenv("SERPER_API_KEY"):
    _tool_instances.insert(0, SerperDevTool())

def get_tools(
    agent_role: str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=_tool_instances
    )