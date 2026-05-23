from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.ml_models import GetModelDetailsTool
from app.services.metrics import GetMetricsForModelTool, DoesModelHaveMetricsTool

_tool_instances = [
    GetModelDetailsTool(),
    GetMetricsForModelTool(),
    DoesModelHaveMetricsTool()
]

tools = {tool.name: tool for tool in _tool_instances}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )