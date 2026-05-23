from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.ml_models import get_ml_models_info
from app.services.metrics import get_metrics

tools = {
    "GetMLModelInfo": get_ml_models_info,
    "GetMLModelMetrics": get_metrics
}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )