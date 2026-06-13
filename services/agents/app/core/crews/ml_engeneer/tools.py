from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.datasets import (
    GetDatasetDetailsTool,
    GetDatasetVersionDetailsTool,
    ListAllDatasetsTool,
)
from app.services.trainer import (
    GetExampleTrainingConfigTool,
    GetAllAvailableModelsTool,
    GetDeviceInfoTool,
    GetOptimizersTool,
    GetSchedulersTool,
    GetMetricsForTrainerTool,
    GetAugmentationsTool,
    ValidateTrainingConfigTool
)

_tool_instances = [
    GetExampleTrainingConfigTool(),
    GetAllAvailableModelsTool(),
    GetDeviceInfoTool(),
    GetOptimizersTool(),
    GetSchedulersTool(),
    GetMetricsForTrainerTool(),
    GetAugmentationsTool(),
    ValidateTrainingConfigTool(),
    GetDatasetDetailsTool(),
    GetDatasetVersionDetailsTool(),
    ListAllDatasetsTool()
]

def get_tools(
    agent_role: str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=_tool_instances
    )