from crewai.tools import tool

from .utils import get_json, handle_errors, get_sample_sizes_for_all_data
from app.logs import get_logger

logger = get_logger(__name__)

@tool("GetDatasetInfo")
@handle_errors
def get_dataset_info(dataset_id: str) -> dict:
    """
    Получить всю информацию о датасете по ID
    
    Args:
        dataset_id: Id датасета
        
    Returns:
        Dict с информацией о датасете или ошибкой
    """    
    return get_json(f"/api/datasets/{dataset_id}")

    
@tool("GetDatasetVersionInfo")
@handle_errors
def get_version_info(
    dataset_id: str,
    version_id: str
) -> dict:
    """
    Получить информацию о версии датасета по ID
    
    Args:
        dataset_id: Id датасета
        version_id: Id версии
        
    Returns:
        Dict с информацией о версии или ошибкой
    """
    return get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")

    
@tool("GetDatasetVersionSplitInfo")
@handle_errors
def get_version_split_info(
    dataset_id: str,
    version_id: str
) -> dict:
    """
    Получить информацию о разбиении датасета (train/val/test)
    
    Args:
        dataset_id: Id датасета
        version_id: Id версии
        
    Returns:
        Dict с информацией о разбиении или ошибкой
    """
    json = get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")
    return get_sample_sizes_for_all_data(json)


@tool("ListDatasets")
@handle_errors
def list_datasets() -> dict:
    """Список всех датасетов"""
    return get_json("/api/datasets/")
