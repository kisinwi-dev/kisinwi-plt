from typing import List, Tuple

from app.logs import get_logger
from app.core.crews.researcher import run_researcher
from app.core.crews.ml_engeneer import MlEngineerResponse, run_ml_engineering

logger = get_logger(__name__)

def reasoning(
    dataset_info: str,
    business_requirements: str,
    denied_hypotheses_info: List[str],
    deployment_constraints: str,
    dataset_id: str,
    dataset_version_id: str,
    verbose: bool = False,
    max_iterations: int = 3
) -> Tuple[MlEngineerResponse, List[str]]:
    """
    Рассуждения агентов и вердикт рассуждений

    Args:
        dataset_info: Информация о датасете
        business_requirements: Требования бизнеса к модели
        deployment_constraints: Наши технические возможности для модели в проде
        dataset_id: ID датасета
        dataset_version_id: ID версии датасета
        max_iterations: Количество попыток исследователя создать предложения удовлетворяющие ML инженера
        verbose: Логирование

    Returns:
        MlEngineerResponse: Ответ ML инженера
        List[str]: Список не верных гипотез
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
            researcher_proposals=researcher_output.to_history_text(),
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            verbose=verbose
        )
        logger.info(f"💾 Решение ML инженера: {'✅ Обучаем' if ml_engineer_output.decision else '🟥 Отказ'}")

        # если решено обучать
        if ml_engineer_output.decision:
            logger.info(f"Решение принято после {iteration + 1} итераций")
            return ml_engineer_output, denied_hypotheses_info

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
    ), denied_hypotheses_info