from typing import List
from crewai.tools import BaseTool

from app.services.metrics import get_metrics
from app.services.ml_models import get_ml_models_info, get_all_ml_models_info
from ..utils import get_tools_with_tracking

tools = {
    "MLModelALLMetrics": get_all_ml_models_info,
    "MLModelMetrics": get_metrics,
    "MLModelInfo": get_ml_models_info
}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )