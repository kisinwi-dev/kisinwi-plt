from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.datasets import (
    GetDatasetDetailsTool,
    GetDatasetVersionDetailsTool,
    GetDatasetSplitSizesTool,
    GetDatasetSplitCountsTool,
    GetDatasetSplitBalanceTool,
    GetDatasetClassDistributionTool,
    GetDatasetImageSizeStatsTool,
    ListAllDatasetsTool
)

_tool_instances = [
    GetDatasetDetailsTool(),
    GetDatasetVersionDetailsTool(),
    GetDatasetSplitSizesTool(),
    GetDatasetSplitCountsTool(),
    GetDatasetSplitBalanceTool(),
    GetDatasetClassDistributionTool(),
    GetDatasetImageSizeStatsTool(),
    ListAllDatasetsTool(),
]

def get_tools(
    agent_role: str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=_tool_instances
    )