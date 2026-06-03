from typing import List
from crewai.tools import BaseTool
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    ArxivPaperTool
)

from ..utils import get_tools_with_tracking

_tool_instances = [
    SerperDevTool(),
    ScrapeWebsiteTool(),
    ArxivPaperTool(),
]

def get_tools(
    agent_role: str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=_tool_instances
    )