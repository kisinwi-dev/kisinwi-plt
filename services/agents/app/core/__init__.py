from typing import List
from pydantic import BaseModel, Field

from app.logs import get_logger
from app.core.crews.researcher import run_researcher
from app.core.crews.ml_engeneer import run_ml_engineering, MlEngineerResponse
from app.core.crews.dataset_analyst import run_dataset_analyst
from app.core.memory import discussion_context, models_context
from app.services.tasker import tasker
from app.services.ml_models import ml_models
from app.services.data import get_dataset_info_classes

logger = get_logger(__name__)

class TrainingOut(BaseModel):
    is_completed_successfully: bool = Field(description="Задача успешно завершена")
    error: str | None = Field(None, description="Информация об ошибке, если задача завершена с ошибкой") 

def development_models(
    dataset_id: str,
    dataset_version_id: str,
    model_name: str,
    deployment_constraints: str,
    business_requirements: str,
    iterations: int = 2,
    verbose: bool = False
):
    """
    Полный цикл разработки моделей

    Args:
        dataset_id: Id датасета
        dataset_version_id: Id версии датасета
        model_name: Как будет называться модель
        deployment_constraints: Наши технические возможности для модели в проде
        business_requirements: Требования бизнеса к модели
        iterations: Количество версий разработанной модели
        verbose: Логирование
    """

    logger.info("Анализа датасета...")
    dataset_analyst_out = run_dataset_analyst(
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        verbose=verbose
    )
    if not dataset_analyst_out.readiness_assessment:
        logger.info("🟥 Анализ датасета показал, что данные не готовы к обучению")
        return None

    logger.info("✅ Анализ датасета")


    for iter in range(1, iterations+1):
        logger.info("Старт рассуждающей группы...")
        ml_engin_out = reasoning(
            dataset_info=dataset_analyst_out.get_short_info(),
            business_requirements=business_requirements,
            denied_hypotheses_info=[],
            verbose=verbose,
            deployment_constraints=deployment_constraints
        )
        if not ml_engin_out.decision:
            logger.info("🟥 МЛ инженер и исследователь не смогли придти к общему мнению.")
            logger.warning("🟥 Обучение остановлено")
            return None
        logger.info("✅ Конец рассуждений.")

        logger.info("Старт секции обучения...")
        trainning_res = training(
            model_name=model_name,
            version_model=iter,
            ml_engin_out=ml_engin_out,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
        )

        if trainning_res.is_completed_successfully:
            logger.info("✅ Модель обучена")
        else:
            logger.info(f"🟥 Модель не была обучена. Ошибка: {trainning_res.error}")

    return None

def training(
    model_name: str,
    version_model: int,
    ml_engin_out: MlEngineerResponse,
    dataset_id: str,
    dataset_version_id: str,
) -> TrainingOut:
    """
    Создание модели на основе ответа мл инженера, запуск обучения и
    ожидание конца тренировки. На выход получаем итоги обучения.

    Args:
        model_name: Имя модели
        version_model: Версия модели
        ml_engin_out: Информация от мл инженера
        dataset_id: ID датасета
        dataset_version_id: ID версии датасета
    """

    if ml_engin_out.ml_model is None:
        return TrainingOut(
            is_completed_successfully=False, 
            error="Не найдены конфигурации для мл модели"
        )

    logger.info("Создание ML модели...")
    new_ml_model = ml_engin_out.ml_model
    model_id = ml_models.create_model(
        name=model_name,
        version=version_model,
        model_type=new_ml_model.type,
        description=new_ml_model.description_model,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        train_params=new_ml_model.configuration,
        classes=get_dataset_info_classes(dataset_id)
    )

    logger.info("✅ ML модель создана")

    logger.info("Создание задачи на обучение...")
    task_id = tasker.task_training_create(
        task_name=f"Обучение модели {model_name} версия {version_model}",
        model_id=model_id,
        discussion_id=discussion_context.get()
    )
    logger.info("✅ Задача создана")

    logger.info("Ожидаем конец выполнения задачи...")
    is_complete, task = tasker.waiting_completed(task_id)

    if is_complete:
        # занесение модели в список обученных моделей
        models_context.add_model(model_id)

    return TrainingOut(
        is_completed_successfully=is_complete,
        error=task["error_message"]
    )

def reasoning(
    dataset_info: str,
    business_requirements: str,
    denied_hypotheses_info: List[str],
    deployment_constraints: str,
    verbose: bool = False,
    max_iterations: int = 3
) -> MlEngineerResponse:
    """
    Рассуждения агентов и вердикт рассуждений

    Args:
        dataset_info: Информация о датасете
        model_name: Как будет называться модель
        business_requirements: Требования бизнеса к модели
        deployment_constraints: Наши технические возможности для модели в проде
        max_iterations: Количество попыток исследователя создать предложения удовлетворяющие ML инженера
        verbose: Логирование
    """
    
    for iteration in range(max_iterations):
        logger.info(f"🔄 Итерация {iteration + 1}/{max_iterations}")

        # генерация гипотез
        logger.info("Генерация гипотез...")
        researcher_output = run_researcher(
            business_requirements=business_requirements,
            dataset_info=dataset_info,
            denied_hypotheses_info=denied_hypotheses_info,
            verbose=verbose
        )
        logger.info("✅ Гипотезы сгенерированы")

        # Проверка гипотез
        logger.info("Проверка гипотез...")
        ml_engineer_output = run_ml_engineering(
            dataset_info=dataset_info,
            business_requirements=business_requirements,
            deployment_constraints=deployment_constraints,
            researcher_proposals=researcher_output.get_full_info(),
            verbose=verbose
        )
        logger.info(f"💾 Решение ML инженера: {'✅ Обучаем' if ml_engineer_output.decision else '🟥 Отказ'}")

        # если решено обучать
        if ml_engineer_output.decision:
            logger.info(f"Решение принято после {iteration + 1} итераций")
            return ml_engineer_output

        # логи
        logger.warning(f"\n\nОтказ на итерации {iteration + 1}: \n {ml_engineer_output.reason}...")
        hypotheses_info = f"Гипотезы: {researcher_output.hypotheses_1} \n {researcher_output.hypotheses_2} \n {researcher_output.hypotheses_3}"
        hypotheses_info += f"\nРешение ML Инженера: {ml_engineer_output.reason}"
        hypotheses_info += f"\nРекомендация: {ml_engineer_output.recommendations}"
        denied_hypotheses_info.append(hypotheses_info)
    
    # Если не удалось найти решение
    logger.error(f"🟥 Не удалось найти решение (пройдено {max_iterations} итераций)")
    return MlEngineerResponse(
        decision=False,
        reason=f"После {max_iterations} попыток не найдено подходящего решения",
        ml_model=None,
        recommendations="Попробуйте изменить требования или обратитесь к специалисту"
    )