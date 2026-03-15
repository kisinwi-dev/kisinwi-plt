from app.core.services.dataset import DatasetManager

def get_dataset_manager() -> DatasetManager:
    """
    Dependency, создаёт DatasetManager для каждого запроса.
    """
    return DatasetManager()