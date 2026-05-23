from typing import List
from crewai.tools import BaseTool

from ..utils import get_tools_with_tracking
from app.services.data import (
    get_dataset_info,
    get_version_info,
    list_datasets,
)

tools = {
    "GetDatasetsList":list_datasets,
    "GetDatasetVersionInfo":get_version_info,
    "GetDatasetInfo":get_dataset_info
}

def get_tools(
    agent_role:str
) -> List[BaseTool]:
    return get_tools_with_tracking(
        agent_role=agent_role,
        tools=tools
    )