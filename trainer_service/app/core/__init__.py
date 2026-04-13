from app.service.tasker import Tasker_Service
from app.service.tasker.shemas import TaskParams
from .task.classification import data, train_model
from .task.classification.models.factory import get_model

from app.logs import get_logger

logger = get_logger(__name__)

def training_model(config: TaskParams):
    """
    Обучение модели.

    Args:
        config: параметры обучения
    """
    try:
        # Загружаем данные
        data_loader_params = config.data_loader_params
        train_loader, val_loader, test_loader, classes = data.load_dataloaders(
            dataset_id=data_loader_params['dataset_id'],
            version_id=data_loader_params['version_id'],
            img_w_size=data_loader_params['img_w_size'],
            img_h_size=data_loader_params['img_h_size'],
            batch_size=data_loader_params['batch_size']
        )

        # Загружаем модель
        model_params = config.model_params
        model = get_model(
            type=model_params["type"],
            num_classes = len(classes),
            pretrained=model_params["pretrained"]
        )

        # Запуск обучения
        trainer_params = config.trainer_params
        trainer = train_model.Trainer(
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

        trainer.train()
    except Exception as e:
        logger.error(e)
        raise