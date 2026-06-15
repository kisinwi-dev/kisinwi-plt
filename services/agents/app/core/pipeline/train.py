import json

from pydantic import BaseModel, Field

from app.core.crews.ml_engeneer import MlEngineerResponse
from app.core.cancellation import raise_if_cancelled
from app.core.memory import discussion_context, models_context
from app.services.agent_history import agent_history_client
from app.services.datasets import get_dataset_info_classes
from app.services.ml_models import ml_models_client
from app.services.tasker import tasker_client
from app.services.utils import parse_in_json
from app.logs import get_logger

logger = get_logger(__name__)


def _apply_authoritative_ids(
    configuration: str | dict,
    dataset_id: str,
    dataset_version_id: str,
) -> dict:
    """
    Принудительно проставить реальные dataset_id/version_id в train-конфиг.

    Trainer загружает данные по `data_loader_params.dataset_id` / `version_id`
    (см. trainer/app/core/datas/loader.py), а эти поля приходят из конфига,
    который сгенерировала LLM. Чтобы id не зависели от того, впишет ли их модель,
    платформа перезаписывает их авторитетными значениями прогона.

    Поднимает исключение, если конфиг не парсится или в нём нет блока
    `data_loader_params` (битый конфиг не должен уйти в обучение).

    Args:
        configuration: конфигурация обучения (строка JSON или dict)
        dataset_id: реальный id датасета прогона
        dataset_version_id: реальный id версии датасета прогона

    Returns:
        dict с гарантированно верными dataset_id/version_id
    """
    config = parse_in_json(configuration)

    data_loader = config.get("data_loader_params")
    if not isinstance(data_loader, dict):
        raise ValueError(
            "В конфигурации обучения нет блока 'data_loader_params' — "
            "невозможно проставить id датасета."
        )

    llm_dataset_id = data_loader.get("dataset_id")
    llm_version_id = data_loader.get("version_id")
    if llm_dataset_id != dataset_id or llm_version_id != dataset_version_id:
        logger.warning(
            "🟧 id датасета в конфиге от ML-инженера не совпали с реальными и были "
            f"перезаписаны. Было: dataset_id={llm_dataset_id!r}, version_id={llm_version_id!r}. "
            f"Стало: dataset_id={dataset_id!r}, version_id={dataset_version_id!r}."
        )
        agent_history_client.warning(
            "Идентификаторы датасета в конфигурации обучения были скорректированы "
            "платформой до фактических значений прогона."
        )

    data_loader["dataset_id"] = dataset_id
    data_loader["version_id"] = dataset_version_id
    return config

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

    # Проставляем реальные id датасета/версии в конфиг — не доверяем тому, что
    # вписала LLM (trainer грузит данные именно по этим полям конфига).
    try:
        train_params = _apply_authoritative_ids(
            ml_model.configuration,
            dataset_id=training_input.dataset_id,
            dataset_version_id=training_input.dataset_version_id,
        )
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"🟥 Некорректная конфигурация обучения от ML-инженера: {e}")
        return TrainingOut(
            is_completed_successfully=False,
            error=f"Некорректная конфигурация обучения: {e}"
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
        train_params=train_params,
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

    version_id = result.get("version_id")
    version = result.get("version")
    if not version_id:
        logger.error("🟥 Сервис ml_models не вернул version_id созданной версии")
        return TrainingOut(
            is_completed_successfully=False,
            error="Не получен version_id созданной версии модели — обучение не запущено"
        )
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

    # Если обучение завершилось из-за остановки всего пайплайна — не продолжаем
    # дальше (в отличие от одиночной отмены обучения), а прерываем пайплайн.
    raise_if_cancelled()

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
