from app.core.train_models_tasks import TrainingTaskManager

training_task_manager = TrainingTaskManager()

async def get_training_task_manager() -> TrainingTaskManager:
    return training_task_manager