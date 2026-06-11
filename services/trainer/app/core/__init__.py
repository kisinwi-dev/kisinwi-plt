from app.api.schemas import TaskParams
from app.service.tasker import tasker_service
from app.service.metrices import MetricesClient
from app.service.ml_models import upload_file_model_in_ml_models
from app.logs import get_logger

from .utils.save_model import save_model_to_onnx
from .datas import create_dataloaders
from .models import get_model
from .trainer import Trainer
from .utils import setup_device

logger = get_logger(__name__)

async def training_model(config: TaskParams, model_id: str):
    """
    Обучение модели.

    Args:
        config: параметры обучения
        model_id: ID модели
    """
    # Проверка технических возможностей
    await tasker_service.update_status_task(percentages=1, status_info="Проверка устройства...")
    device = setup_device(config.device)
    await tasker_service.update_status_task(percentages=2, status_info="Устройство проверено на вычислительные возможности.")

    # Загружаем данные
    await tasker_service.update_status_task(percentages=3, status_info="Загрузка данных...")
    train_loader, val_loader, test_loader, classes = create_dataloaders(config.data_loader_params)
    await tasker_service.update_status_task(percentages=5, status_info="Данные загружены.")

    # Загружаем модель
    await tasker_service.update_status_task(percentages=6, status_info="Загрузка модели...")
    model = get_model(
        config.model_params,
        num_classes = len(classes)
    ).to(device)
    await tasker_service.update_status_task(percentages=10, status_info=f"Модель {config.model_params.type} загружена.")

    await tasker_service.update_status_task(percentages=11, status_info="Настройка метрик...")
    metric_client = MetricesClient(
        model_id=model_id,
        metrices_params=config.metrices_params,
        classes=classes,
        device=device,
        early_stop_params=config.trainer_params.early_stop
    )
    await tasker_service.update_status_task(percentages=12, status_info="Метрики настроены.")

    # Запуск обучения
    await tasker_service.update_status_task(percentages=13, status_info="Формирование процесса обучения...")
    trainer = Trainer(
        # ID модели
        model_id=model_id,
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
    await tasker_service.update_status_task(percentages=19, status_info="Процесс обучения сформирован")
    await tasker_service.update_status_task(percentages=20, status_info="Обучение...")
    model = await trainer.train(20, 80)
    await tasker_service.update_status_task(percentages=80, status_info="Модель обучена.")

    await tasker_service.update_status_task(percentages=81, status_info="Сохранение модели...")
    sample_img = train_loader.dataset[0][0]
    input_shape = (1,) + sample_img.shape
    onnx_path = await save_model_to_onnx(
        model_id=model_id,
        model=model,
        input_shape=input_shape,
        device=device,
        classes=classes
    )
    await tasker_service.update_status_task(percentages=88, status_info="Сохранение модели...")
    await upload_file_model_in_ml_models(model_id, onnx_path, "onnx_model")
    await tasker_service.update_status_task(percentages=95, status_info="Модель сохранена.")

    # Модель выгружена — чекпоинт лучшей эпохи больше не нужен
    trainer.checkpoint_path.unlink(missing_ok=True)
