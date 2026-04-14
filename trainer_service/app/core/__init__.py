from app.service.tasker import tasker_service, TaskParams
from app.logs import get_logger

from .datas import create_dataloaders
from .models import get_model
from .train_model import Trainer

logger = get_logger(__name__)

async def training_model(config: TaskParams):
    """
    Обучение модели.

    Args:
        config: параметры обучения
    """
    try:
        # Загружаем данные
        await tasker_service.update_status_task(1, description="Загрузка данных...")
        data_loader_params = config.data_loader_params
        train_loader, val_loader, test_loader, classes = create_dataloaders(data_loader_params)
        await tasker_service.update_status_task(5, description="Данные загружены.")

        # Загружаем модель
        await tasker_service.update_status_task(6, description="Загрузка модели...")
        model_params = config.model_params
        model = get_model(
            config.model_params,
            num_classes = len(classes)
        )
        await tasker_service.update_status_task(10, description=f"Модель {model_params.type} загружена.")

        # Запуск обучения
        await tasker_service.update_status_task(11, description=f"Формирование процесса обучения...")
        trainer_params = config.trainer_params
        trainer = Trainer(
            tasker_service,
            model,
            train_loader,
            val_loader,
            test_loader,
            classes,
            loss_fn_config=trainer_params["loss_fn_config"],
            optimizer_config=trainer_params["optimizer_config"],
            scheduler_config=trainer_params["scheduler_config"],
            device=trainer_params["device"],
            epochs=1 # trainer_params["epochs"],
        )
        await tasker_service.update_status_task(12, description=f"Процесса обучения сформирован")
        await tasker_service.update_status_task(13, description=f"Обучения...")
        trainer.train()

    except Exception as e:
        logger.error(e)
        raise