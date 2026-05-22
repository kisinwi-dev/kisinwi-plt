from typing import List
from crewai.tools import BaseTool
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    ArxivPaperTool
)

from ..utils import get_tools_with_tracking

tools = {
    "SerperDevTool": SerperDevTool(),
    "ScrapeWebsiteTool": ScrapeWebsiteTool(),
    "ArxivPaperTool": ArxivPaperTool()
}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )