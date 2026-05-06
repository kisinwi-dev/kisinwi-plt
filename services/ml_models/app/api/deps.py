from app.core.train_models_tasks import MlModelsManager
from app.core.files import FilesManager

ml_models_manager = MlModelsManager()
files_manager = FilesManager()

async def get_ml_models_manager() -> MlModelsManager:
    return ml_models_manager

async def get_files_manager() -> FilesManager:
    return files_manager