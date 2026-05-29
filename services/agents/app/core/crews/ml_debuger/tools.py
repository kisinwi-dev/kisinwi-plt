from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.datasets import GetDatasetDetailsTool, ListAllDatasetsTool
from app.services.trainer import (
    GetExampleTrainingConfigTool,
    GetAllAvailableModelsTool,
    GetDeviceInfoTool,
    GetOptimizersTool,
    GetSchedulersTool,
    GetMetricsForTrainerTool
)

_tool_instances = [
    GetExampleTrainingConfigTool(),
    GetAllAvailableModelsTool(),
    GetDeviceInfoTool(),
    GetOptimizersTool(),
    GetSchedulersTool(),
    GetMetricsForTrainerTool(),
    GetDatasetDetailsTool(),
    ListAllDatasetsTool()
]

tools = {tool.name: tool for tool in _tool_instances}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )