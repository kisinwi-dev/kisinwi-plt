from typing import List

from app.logs import get_logger
from app.core.crews.dataset_analyst import run_dataset_analyst
from app.core.crews.reporter import run_reporter
from app.core.memory import iteration_context
from app.services.agent_history import agent_history_client
from .pipeline import (
    train_and_debug, reasoning,
    TrainingInput
)

logger = get_logger(__name__)

def development_models(
    dataset_id: str,
    dataset_version_id: str,
    model_name: str,
    deployment_constraints: str,
    business_requirements: str,
    denied_hypotheses_info: List[str] = [],
    max_iter: int = 2,
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
        denied_hypotheses_info: Какие гипотезы стоит откинуть сразу (опицонально)
        max_iter: Количество версий разработанной модели (опицонально)
        verbose: Логирование (опицонально)
    """

    logger.info("Анализа датасета...")
    agent_history_client.info("Запуск пайплайна разработки модели. Анализ датасета...")
    dataset_analyst_out = run_dataset_analyst(
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        verbose=verbose
    )
    if not dataset_analyst_out.readiness_assessment:
        logger.info("🟥 Анализ датасета показал, что данные не готовы к обучению")
        agent_history_client.error("Анализ датасета показал, что данные не готовы к обучению. Пайплайн остановлен.")
        return None

    logger.info("✅ Анализ датасета")
    agent_history_client.info("Анализ датасета завершён: данные готовы к обучению.")
    version_model=0

    for iter in range(1, max_iter+1):
        iteration_context.set(iter)
        version_model+=1

        info_start_iter = f"Полный цикл обучения №{iter} из {max_iter}"
        agent_history_client.info(f"{info_start_iter}. Этап рассуждения агентов.")
        logger.info(
            f"\n{'='*100}"
            f"\n{'Этап рассуждения':^100}"
            f"\n{info_start_iter:^100}"
            f"\n{'='*100}"
        )
        # Старт рассуждений агентов над задачей
        # Получаем ответ ML инженера и список не верных гипотезы
        ml_engin_out, denied_hypotheses_info = reasoning(
            dataset_info=dataset_analyst_out.to_history_text(),
            business_requirements=business_requirements,
            denied_hypotheses_info=denied_hypotheses_info,
            verbose=verbose,
            deployment_constraints=deployment_constraints,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id
        )
        if not ml_engin_out.decision:
            logger.info("🟥 МЛ инженер и исследователь не смогли придти к общему мнению.")
            logger.warning("🟥 Обучение остановлено")
            agent_history_client.error(
                "ML-инженер и исследователь не пришли к общему мнению. Обучение остановлено."
            )
            return None
        logger.info("✅ Конец рассуждений.")
        agent_history_client.info("Рассуждение завершено: конфигурация обучения согласована.")

        logger.info(
            f"\n{'='*100}"
            f"\n{'Этап Обучения':^100}"
            f"\n{info_start_iter:^100}"
            f"\n{'='*100}"
        )
        agent_history_client.info(f"{info_start_iter}. Запуск обучения модели.")
        # Создаём экземпляр модели для запуска тренировок
        training_input=TrainingInput(
            model_name=model_name,
            version_model=version_model,
            ml_engin_out=ml_engin_out,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id
        )
        # Запуск процеса тренировки 
        logger.info("")
        training_res, version_model = train_and_debug(training_input)

        info_final_iter_1_line = f"Полный цикл обучения №{iter} из {max_iter}"
        info_final_iter_2_line =  "завершился с результатом:"
        if training_res.is_completed_successfully:
            logger.info(
                f"\n{'='*100}"
                f"\n{info_final_iter_1_line:^100}"
                f"\n{info_final_iter_2_line:^100}"
                f"\n{'✅ Модель успешно обучена':^100}"
                f"\n{'='*100}"
            )
            agent_history_client.info(f"Цикл обучения №{iter} из {max_iter} завершён: модель успешно обучена.")

        else:
            logger.info(
                f"\n{'='*100}"
                f"\n{info_final_iter_1_line:^100}"
                f"\n{info_final_iter_2_line:^100}"
                f"\n{'🟥 Не удалось обучить модель':^100}"
                f"\nОписание: {training_res.error}"
                f"\n{'='*100}"
            )
            agent_history_client.warning(
                f"Цикл обучения №{iter} из {max_iter} провалился: {training_res.error}"
            )
            # Сообщаем следующей итерации о провале обучения, чтобы исследователь и
            # ML инженер не повторили то же решение.
            ml_model = ml_engin_out.ml_model
            what_trained = f"{ml_model.type} — {ml_model.description_model}" if ml_model else "модель"
            denied_hypotheses_info.append(
                "Предыдущая попытка обучения провалилась и не была исправлена дебагером.\n"
                f"Что обучали: {what_trained}\n"
                f"Ошибка обучения: {training_res.error}\n"
                "Учти это и предложи другое решение."
            )

    # Подводим итоги обучений
    agent_history_client.info("Все циклы обучения завершены. Формирование итогового отчёта...")
    result = run_reporter(
        business_requirements=business_requirements,
        deployment_constraints=deployment_constraints,
        verbose=verbose
    )

    iteration_context.clear()
    return result

