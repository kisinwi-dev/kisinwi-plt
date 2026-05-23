from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.trainer import get_example_run_config_trainer_json
from app.core.crews.ml_models_searcher.ml_models_searcher import tool_run_ml_models_searcher
from app.core.crews.praxis_searcher.praxis_searcher import tool_run_praxis_searcher

tools = {
    "UsePraxisSearcher": tool_run_praxis_searcher,
    "UseMLModelsSearcher": tool_run_ml_models_searcher,
    "GetExampleTrainingConfig": get_example_run_config_trainer_json
}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )