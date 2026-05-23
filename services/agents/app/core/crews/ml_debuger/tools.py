from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.data import get_dataset_info, list_datasets
from app.services.trainer import (
    get_example_run_config_trainer_json,
    get_type_and_name_models,
    get_info_device,
    get_scheduler,
    get_optimizers,
    get_metrics
)

tools = {
    "GetExampleTrainingConfig": get_example_run_config_trainer_json,
    "GetConfigMLModels": get_type_and_name_models,
    "GetConfigInfoDevice": get_info_device,
    "GetConfigScheduler":get_scheduler,
    "GetConfigOptimizers":get_optimizers,
    "GetConfigMetrics":get_metrics,
    "GetDatasetInfo":get_dataset_info,
    "GetDatasetsList":list_datasets
}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )