from typing import List
from crewai.tools import BaseTool
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    ArxivPaperTool
)

from ..utils import get_tools_with_tracking

tools = {
    "Search the internet with Serper": SerperDevTool(),
    "ScrapeWebsiteTool": ScrapeWebsiteTool(),
    "Arxiv Paper Fetcher and Download": ArxivPaperTool()
}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )