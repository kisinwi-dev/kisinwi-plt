from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.data import (
    GetDatasetDetailsTool,
    GetDatasetVersionDetailsTool,
    GetDatasetSplitSizesTool,
    ListAllDatasetsTool
)

_tool_instances = [
    GetDatasetDetailsTool(),
    GetDatasetVersionDetailsTool(),
    GetDatasetSplitSizesTool(),
    ListAllDatasetsTool(),
]

tools = {tool.name: tool for tool in _tool_instances}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )