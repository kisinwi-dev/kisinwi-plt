import json

from app.logs import get_logger
from app.core.cancellation import raise_if_cancelled
from app.core.crews.ml_engeneer_quick import run_quick_ml_engineering
from app.core.crews.metrics_analyst import run_metrics_analyst
from app.core.defaults import DEFAULT_BUSINESS_REQUIREMENTS, DEFAULT_DEPLOYMENT_CONSTRAINTS
from app.services.agent_history import agent_history_client
from app.services.datasets import get_dataset_details, get_dataset_version_details
from app.services.ml_models import load_model_history
from .pipeline import training, TrainingInput

logger = get_logger(__name__)


def _build_dataset_info(dataset_id: str, dataset_version_id: str) -> str:
    """Сырые метаданные датасета и версии вместо ответа Dataset Analyst."""
    dataset = get_dataset_details(dataset_id)
    version = get_dataset_version_details(dataset_id, dataset_version_id)
    return (
        "## Метаданные датасета (сырые, без анализа)\n\n"
        f"**Датасет:**\n```json\n{json.dumps(dataset, ensure_ascii=False, indent=2)}\n```\n\n"
        f"**Версия:**\n```json\n{json.dumps(version, ensure_ascii=False, indent=2)}\n```"
    )


def quick_training_models(
    dataset_id: str,
    dataset_version_id: str,
    model_name: str,
    deployment_constraints: str | None = None,
    business_requirements: str | None = None,
    model_id: str | None = None,
    verbose: bool = False
):
    """
    Быстрый пайплайн: ML-инженер + аналитик метрик

    Один проход без итераций и дебага: ML-инженер составляет конфигурацию
    обучения по сырым метаданным датасета, запускается обучение, после чего
    аналитик метрик разбирает результат.

    Args:
        dataset_id: Id датасета
        dataset_version_id: Id версии датасета
        model_name: Как будет называться модель
        deployment_constraints: Наши технические возможности для модели в проде
            (опционально; без них агенты минимизируют затраты сами)
        business_requirements: Требования бизнеса к модели
            (опционально; без них агенты максимизируют качество сами)
        model_id: ID существующей модели — новые версии создаются под ней (опицонально)
        verbose: Логирование (опицонально)
    """
    business_requirements = (business_requirements or "").strip() or DEFAULT_BUSINESS_REQUIREMENTS
    deployment_constraints = (deployment_constraints or "").strip() or DEFAULT_DEPLOYMENT_CONSTRAINTS

    raise_if_cancelled()
    logger.info("Получение метаданных датасета...")
    agent_history_client.info(
        "Запуск быстрого пайплайна (ML-инженер + аналитик метрик). Получение метаданных датасета..."
    )
    dataset_info = _build_dataset_info(dataset_id, dataset_version_id)

    if model_id is not None:
        logger.info("Получение истории версий модели...")
        agent_history_client.info(
            "Продолжаем обучение существующей модели. Получение истории версий..."
        )
    model_history = load_model_history(model_id)
    if model_history is None:
        logger.error(f"🟥 Модель {model_id} не найдена в реестре")
        agent_history_client.error(
            f"Модель {model_id} не найдена в реестре. Пайплайн остановлен."
        )
        return None

    agent_history_client.info("Метаданные получены. ML-инженер составляет конфигурацию обучения...")
    ml_engin_out = run_quick_ml_engineering(
        dataset_info=dataset_info,
        business_requirements=business_requirements,
        deployment_constraints=deployment_constraints,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        model_history=model_history,
        verbose=verbose
    )
    if not ml_engin_out.decision:
        # В быстром прогоне отказ запрещён по качеству — decision=false означает,
        # что обучать физически нечего (датасет/версия не найдены).
        logger.info("🟥 Быстрый прогон: обучение невозможно (нет данных).")
        agent_history_client.error(
            f"Обучение невозможно: датасет или версия не найдены. {ml_engin_out.reason}"
        )
        return None
    logger.info("✅ ML-инженер составил конфигурацию обучения.")
    agent_history_client.info("Конфигурация обучения готова. Запуск обучения модели.")

    training_res = training(TrainingInput(
        model_name=model_name,
        model_id=model_id,
        ml_engin_out=ml_engin_out,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id
    ))
    if training_res.is_cancelled:
        # Человек остановил обучение — это не сбой. Разбираем частичные метрики
        # отменённой версии и возвращаем анализ (а не помечаем как провал).
        logger.info("🟦 Обучение остановлено пользователем. Разбор частичных метрик...")
        agent_history_client.warning(
            "Обучение остановлено пользователем. Разбираем частичные метрики на момент остановки."
        )
        if training_res.version_id is None:
            agent_history_client.info("Метрики на момент остановки недоступны. Быстрый пайплайн завершён.")
            return None
        analysis = run_metrics_analyst(
            model_id=training_res.version_id,
            business_goal=business_requirements,
            verbose=verbose
        )
        agent_history_client.info("Быстрый пайплайн завершён (обучение остановлено пользователем).")
        return analysis.to_history_text()

    if not training_res.is_completed_successfully:
        logger.info(f"🟥 Обучение провалилось: {training_res.error}")
        agent_history_client.error(f"Обучение провалилось: {training_res.error}")
        return None
    logger.info("✅ Модель успешно обучена.")
    agent_history_client.info("Модель успешно обучена. Анализ метрик...")

    trained_version_id = training_res.version_id
    analysis = run_metrics_analyst(
        model_id=trained_version_id,
        business_goal=business_requirements,
        verbose=verbose
    )

    agent_history_client.info("Быстрый пайплайн завершён.")
    return analysis.to_history_text()
