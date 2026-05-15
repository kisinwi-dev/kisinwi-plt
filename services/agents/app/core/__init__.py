from typing import List

from app.logs import get_logger
from app.core.crews.researcher import run_researcher
from app.core.crews.ml_engeneer import run_ml_engineering, MlEngineerResponse
from app.core.crews.dataset_analyst import run_dataset_analyst

from .pipeline import training, debuging, TrainingOut, TrainingInput

logger = get_logger(__name__)

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
    version_model=0

    for iter in range(1, iterations+1):
        version_model+=1

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

        training_input=TrainingInput(
            model_name=model_name,
            version_model=version_model,
            ml_engin_out=ml_engin_out,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id
        )
        
        # внутри функций training и debuging реализовано логирование        
        training_res = training(training_input)

        training_res = TrainingOut(
            is_completed_successfully=False,
            error="Ошибочка"
        )

        training_res, version_model = debuging(training_input, training_res)

        if training_res.is_completed_successfully:
            logger.info("✅ Модель успешно обучена")
        else:
            logger.info(
                "🟥 Не удалось обучить модель"
                f"\nОписание: {training_res.error}"
            )

    return None

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