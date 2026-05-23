from typing import List
from crewai.tools import BaseTool

from app.services.metrics import GetMetricsForModelTool, DoesModelHaveMetricsTool
from app.services.ml_models import GetModelDetailsTool, GetMultipleModelsDetailsTool
from ..utils import get_tools_with_tracking

_tool_instances = [
    GetMultipleModelsDetailsTool(),
    GetModelDetailsTool(),
    DoesModelHaveMetricsTool(),
    GetMetricsForModelTool()
]

tools = {tool.name: tool for tool in _tool_instances}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )