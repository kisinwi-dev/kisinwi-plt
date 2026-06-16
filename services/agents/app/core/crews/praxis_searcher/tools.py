import os
from typing import List
from crewai.tools import BaseTool
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    ArxivPaperTool
)

from ..utils import get_tools_with_tracking

_tool_instances: List[BaseTool] = [
    ScrapeWebsiteTool(),
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