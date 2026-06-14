from pydantic import BaseModel, Field

from app.core.crews.ml_engeneer import MlEngineerResponse
from app.core.memory import discussion_context, models_context
from app.services.agent_history import agent_history_client
from app.services.datasets import get_dataset_info_classes
from app.services.ml_models import ml_models_client
from app.services.tasker import tasker_client
from app.logs import get_logger

logger = get_logger(__name__)

class TrainingOut(BaseModel):
    is_completed_successfully: bool = Field(description="Задача успешно завершена")
    is_cancelled: bool = Field(False, description="Обучение остановлено пользователем (не ошибка)")
    version_id: str | None = Field(None, description="ID версии модели, по которой шло обучение")
    error: str | None = Field(None, description="Информация об ошибке, если задача завершена с ошибкой")

class TrainingInput(BaseModel):
    model_name: str = Field(description="Имя модели")
    model_id: str | None = Field(None, description="ID существующей модели для продолжения обучения")
    dataset_id: str = Field(description="ID датасета")
    dataset_version_id: str = Field(description="ID версии датасета")
    ml_engin_out: MlEngineerResponse = Field(description="Полученная информация от ML инженера")

def training(
    training_input: TrainingInput
) -> TrainingOut:
    """
    Создание версии модели на основе ответа мл инженера, запуск обучения и
    ожидание конца тренировки. На выход получаем итоги обучения.

    Args:
        training_input: вся информация позволяющая запустить обучение

    * Всё логирование о запуске тренировок и итогах тренировки выполняется
    внутри функции
    """

    ml_model = training_input.ml_engin_out.ml_model

    if ml_model is None:
        return TrainingOut(
            is_completed_successfully=False,
            error="Не найдены конфигурации для мл модели"
        )

    logger.info("Создание версии ML модели...")
    result = ml_models_client.create_model_version(
        # информация о модели
        name=training_input.model_name,
        model_type=ml_model.type,
        description=ml_model.description_model,
        # информация о данных
        dataset_id=training_input.dataset_id,
        dataset_version_id=training_input.dataset_version_id,
        train_params=ml_model.configuration,
        classes=get_dataset_info_classes(training_input.dataset_id),
        model_id=training_input.model_id
    )

    # При ошибке create_model_version возвращает {"ERROR": ...}. Не отправляем
    # обучение с битым id версии — сразу сообщаем о провале.
    if "ERROR" in result:
        error = result["ERROR"]
        logger.error(f"🟥 Не удалось создать версию ML модели: {error}")
        return TrainingOut(
            is_completed_successfully=False,
            error=f"Не удалось создать версию ML модели: {error}"
        )

    version_id = result["version_id"]
    version = result["version"]
    logger.info(f"✅ Создана версия {version} ML модели")

    logger.info("Создание задачи на обучение...")
    task_id = tasker_client.task_training_create(
        task_name=f"Обучение модели {training_input.model_name} версия {version}",
        model_id=version_id,
        discussion_id=discussion_context.get()
    )
    logger.info("✅ Задача создана")
    agent_history_client.info(
        f"Запущено обучение модели «{training_input.model_name}» (версия {version})"
    )

    logger.info("Ожидаем конец выполнения задачи...")
    is_complete, task = tasker_client.waiting_completed(task_id)

    if is_complete:
        # занесение версии в список обученных моделей
        logger.info("✅ Модель обучена")
        agent_history_client.info(
            f"Обучение модели «{training_input.model_name}» (версия {version}) завершено успешно."
        )
        models_context.add_model(version_id)
        return TrainingOut(
            is_completed_successfully=True,
            version_id=version_id,
            error=task["error_message"]
        )

    # При cancelled error_message пустой — причина лежит в status_info
    error = task["error_message"] or task.get("status_info")

    if task.get("status") == "cancelled":
        # Это не сбой обучения, а сознательная остановка человеком. Отменённую
        # версию не кладём в models_context (она не готовая модель), но version_id
        # прокидываем — по нему потом достаём частичные метрики.
        logger.info(f"🟦 Обучение остановлено пользователем. Причина: {error}")
        agent_history_client.warning(
            f"Обучение модели «{training_input.model_name}» (версия {version}) "
            f"остановлено пользователем. {error or ''}".strip()
        )
        return TrainingOut(
            is_completed_successfully=False,
            is_cancelled=True,
            version_id=version_id,
            error=error
        )

    logger.info(
        "🟥 Модель не была обучена."
        f"\nПроизошла ошибка в процесе обучения. Причина: {error}"
    )
    agent_history_client.error(
        f"Обучение модели «{training_input.model_name}» (версия {version}) "
        f"не завершено. Причина: {error}"
    )
    return TrainingOut(
        is_completed_successfully=False,
        version_id=version_id,
        error=error
    )
