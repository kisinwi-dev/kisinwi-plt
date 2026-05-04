from app.core.train_models_tasks import MlModelsManager

ml_models_manager = MlModelsManager()

async def get_ml_models_manager() -> MlModelsManager:
    return ml_models_manager