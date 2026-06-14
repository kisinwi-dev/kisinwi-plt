from app.api.schemas import TaskParams
from app.service.tasker import tasker_service
from app.service.metrices import MetricesClient, send_checkpoint_info
from app.service.ml_models import upload_file_model_in_ml_models
from app.logs import get_logger

from .utils.save_model import save_model_to_onnx
from .datas import create_dataloaders
from .models import get_model, predownload_weights
from .trainer import Trainer
from .utils import setup_device

logger = get_logger(__name__)

# Прогресс задачи по этапам пайплайна:
# 0–19 подготовка, 20–80 эпохи обучения, 81–95 сохранение, 100 готово (ставит воркер)
PROGRESS_DEVICE_CHECK = 1
PROGRESS_DEVICE_READY = 2
PROGRESS_DATA_LOADING = 3
PROGRESS_DATA_READY = 5
PROGRESS_WEIGHTS_DOWNLOAD = 6
PROGRESS_WEIGHTS_READY = 8
PROGRESS_MODEL_LOADING = 9
PROGRESS_MODEL_READY = 10
PROGRESS_METRICS_SETUP = 11
PROGRESS_METRICS_READY = 12
PROGRESS_TRAINER_SETUP = 13
PROGRESS_TRAINER_READY = 19
PROGRESS_TRAINING_START = 20
PROGRESS_TRAINING_END = 80
PROGRESS_SAVING = 81
PROGRESS_UPLOADING = 88
PROGRESS_SAVED = 95

async def training_model(config: TaskParams, model_id: str):
    """
    Обучение модели.

    Args:
        config: параметры обучения
        model_id: ID модели
    """
    # Проверка технических возможностей
    await tasker_service.update_status_task(percentages=PROGRESS_DEVICE_CHECK, status_info="Проверка устройства...")
    device = setup_device(config.device)
    await tasker_service.update_status_task(percentages=PROGRESS_DEVICE_READY, status_info="Устройство проверено на вычислительные возможности.")

    # Загружаем данные
    await tasker_service.update_status_task(percentages=PROGRESS_DATA_LOADING, status_info="Загрузка данных...")
    train_loader, val_loader, test_loader, classes = create_dataloaders(config.data_loader_params)
    await tasker_service.update_status_task(percentages=PROGRESS_DATA_READY, status_info="Данные загружены.")

    # Предзагрузка весов pretrained-модели в кэш (отдельный этап с видимым прогрессом)
    if config.model_params.pretrained:
        await predownload_weights(
            model_name=config.model_params.type,
            tasker_service=tasker_service,
            progress_start=PROGRESS_WEIGHTS_DOWNLOAD,
            progress_end=PROGRESS_WEIGHTS_READY,
        )

    # Загружаем модель (веса берутся из кэша мгновенно)
    await tasker_service.update_status_task(percentages=PROGRESS_MODEL_LOADING, status_info="Загрузка модели...")
    model = get_model(
        config.model_params,
        num_classes = len(classes)
    ).to(device)
    await tasker_service.update_status_task(percentages=PROGRESS_MODEL_READY, status_info=f"Модель {config.model_params.type} загружена.")

    await tasker_service.update_status_task(percentages=PROGRESS_METRICS_SETUP, status_info="Настройка метрик...")
    metric_client = MetricesClient(
        model_id=model_id,
        classes=classes,
        device=device,
        early_stop_params=config.trainer_params.early_stop
    )
    await tasker_service.update_status_task(percentages=PROGRESS_METRICS_READY, status_info="Метрики настроены.")

    # Останавливаем фоновый поток метрик и при успехе, и при ошибке/отмене
    try:
        # Запуск обучения
        await tasker_service.update_status_task(percentages=PROGRESS_TRAINER_SETUP, status_info="Формирование процесса обучения...")
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
        await tasker_service.update_status_task(percentages=PROGRESS_TRAINER_READY, status_info="Процесс обучения сформирован")
        await tasker_service.update_status_task(percentages=PROGRESS_TRAINING_START, status_info="Обучение...")
        model = await trainer.train(PROGRESS_TRAINING_START, PROGRESS_TRAINING_END)
        await tasker_service.update_status_task(percentages=PROGRESS_TRAINING_END, status_info="Модель обучена.")
        # Инфа о сохранённых весах: лучшая эпоха по early-stop-метрике,
        # либо финальная, если улучшение не фиксировалось (best_value = None)
        await send_checkpoint_info(
            model_id=model_id,
            epoch=trainer.best_epoch or trainer.last_epoch,
            metric=config.trainer_params.early_stop.metric_name,
            value=trainer.best_value,
        )
    finally:
        metric_client.close()

    await tasker_service.update_status_task(percentages=PROGRESS_SAVING, status_info="Сохранение модели...")
    sample_img = train_loader.dataset[0][0]
    input_shape = (1,) + sample_img.shape
    onnx_path = await save_model_to_onnx(
        model_id=model_id,
        model=model,
        input_shape=input_shape,
        device=device,
        classes=classes
    )
    await tasker_service.update_status_task(percentages=PROGRESS_UPLOADING, status_info="Сохранение модели...")
    await upload_file_model_in_ml_models(model_id, onnx_path, "onnx_model")
    await tasker_service.update_status_task(percentages=PROGRESS_SAVED, status_info="Модель сохранена.")

    # Модель выгружена — чекпоинт лучшей эпохи больше не нужен
    trainer.checkpoint_path.unlink(missing_ok=True)
