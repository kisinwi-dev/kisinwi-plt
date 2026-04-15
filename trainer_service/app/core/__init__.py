from app.api.schemes import TaskParams
from app.service.tasker import tasker_service
from app.service.metrices import MetricesClient
from app.logs import get_logger

from .datas import create_dataloaders
from .models import get_model
from .trainer import Trainer
from .utils import setup_device

logger = get_logger(__name__)

async def training_model(config: TaskParams):
    """
    Обучение модели.

    Args:
        config: параметры обучения
    """
    try:

        # Проверка технических возможностей
        await tasker_service.update_status_task(1, description="Проверка устройства...")
        device = setup_device(config.device)
        await tasker_service.update_status_task(2, description="Устройство проверено на вычислительные возможности.")

        # Загружаем данные
        await tasker_service.update_status_task(3, description="Загрузка данных...")
        train_loader, val_loader, test_loader, classes = create_dataloaders(config.data_loader_params)
        await tasker_service.update_status_task(5, description="Данные загружены.")

        # Загружаем модель
        await tasker_service.update_status_task(6, description="Загрузка модели...")
        model = get_model(
            config.model_params,
            num_classes = len(classes)
        ).to(device)
        await tasker_service.update_status_task(10, description=f"Модель {config.model_params.type} загружена.")

        await tasker_service.update_status_task(11, description="Настройка метрик...")
        metric_client = MetricesClient(
            task_id=tasker_service.task_id,
            metrices_params=config.metrices_params,
            num_class=len(classes),
            device=device
        )
        await tasker_service.update_status_task(12, description="Метрики настроены.")

        # Запуск обучения
        await tasker_service.update_status_task(13, description=f"Формирование процесса обучения...")
        trainer = Trainer(
            # Вспомогательные сервисы
            tasker_service=tasker_service,
            metric_service=metric_client,
            # Модель
            model=model,
            # Данные
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            classes=classes,
            # Устройство
            device=device,
            # Конфигурация
            train_params=config.trainer_params
        )
        await tasker_service.update_status_task(19, description=f"Процесса обучения сформирован")
        await tasker_service.update_status_task(20, description=f"Обучения...")
        model = await trainer.train(20, 80)
        await tasker_service.update_status_task(80, description=f"Модель обучена.")


    except Exception as e:
        logger.error(e)
        raise