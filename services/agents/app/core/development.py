from typing import List

from app.logs import get_logger
from app.core.crews.dataset_analyst import run_dataset_analyst
from app.core.crews.metrics_analyst import run_metrics_analyst
from app.core.crews.reporter import run_reporter
from app.core.defaults import DEFAULT_BUSINESS_REQUIREMENTS, DEFAULT_DEPLOYMENT_CONSTRAINTS
from app.core.memory import iteration_context
from app.services.agent_history import agent_history_client
from app.services.ml_models import load_model_history
from .pipeline import (
    train_and_debug, reasoning,
    TrainingInput
)

logger = get_logger(__name__)

# ponytail: предохранитель в авто-режиме (max_iter=0), поднять если мало
AUTO_ATTEMPTS_CAP = 5

def development_models(
    dataset_id: str,
    dataset_version_id: str,
    model_name: str,
    deployment_constraints: str | None = None,
    business_requirements: str | None = None,
    denied_hypotheses_info: List[str] = [],
    max_iter: int = 0,
    model_id: str | None = None,
    skip_dataset_check: bool = False,
    verbose: bool = False
):
    """
    Полный цикл разработки моделей

    Args:
        dataset_id: Id датасета
        dataset_version_id: Id версии датасета
        model_name: Как будет называться модель
        deployment_constraints: Наши технические возможности для модели в проде
            (опционально; без них агенты минимизируют затраты сами)
        business_requirements: Требования бизнеса к модели
            (опционально; без них агенты максимизируют качество сами)
        denied_hypotheses_info: Какие гипотезы стоит откинуть сразу (опицонально)
        max_iter: Количество попыток обучения. >0 — ровно столько попыток (без
            досрочного стопа). 0 (по умолчанию) — агент сам определяет количество:
            цикл идёт до достижения требований, но не больше AUTO_ATTEMPTS_CAP.
        model_id: ID существующей модели — новые версии создаются под ней (опицонально)
        skip_dataset_check: Продолжать обучение, даже если аналитик данных забраковал
            датасет (readiness_assessment=False). По умолчанию False.
        verbose: Логирование (опицонально)
    """
    business_requirements = (business_requirements or "").strip() or DEFAULT_BUSINESS_REQUIREMENTS
    deployment_constraints = (deployment_constraints or "").strip() or DEFAULT_DEPLOYMENT_CONSTRAINTS

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

    logger.info("Анализа датасета...")
    agent_history_client.info("Запуск пайплайна разработки модели. Анализ датасета...")
    dataset_analyst_out = run_dataset_analyst(
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        verbose=verbose
    )
    dataset_info = dataset_analyst_out.to_history_text()
    if not dataset_analyst_out.readiness_assessment:
        if not skip_dataset_check:
            logger.info("🟥 Анализ датасета показал, что данные не готовы к обучению")
            agent_history_client.error("Анализ датасета показал, что данные не готовы к обучению. Пайплайн остановлен.")
            return None
        logger.warning("⚠️ Датасет не готов к обучению, но skip_dataset_check=True — продолжаем")
        agent_history_client.warning(
            "Аналитик данных забраковал датасет, но проверка отключена при запуске. "
            "Продолжаем обучение на свой риск."
        )
        # Явный сигнал downstream-агентам (researcher, ml_engineer): датасет
        # забракован, но пользователь настоял именно на нём — не отказываться от
        # обучения, а компенсировать изъяны подбором конфигурации.
        dataset_info += (
            "\n\n## ⚠️ Решение пользователя: обучать на этом датасете\n"
            "Аналитик данных признал датасет НЕ готовым к обучению (проблемы выше). "
            "Пользователь осознанно настоял на обучении именно на этом датасете и принимает риски. "
            "НЕ отказывайтесь от обучения из-за качества данных вместо этого подберите "
            "конфигурацию, которая максимально компенсирует выявленные изъяны "
            "(аугментация, регуляризация, борьба с дисбалансом классов, очистка на лету и т.п.)."
        )
    else:
        logger.info("✅ Анализ датасета")
        agent_history_client.info("Анализ датасета завершён: данные готовы к обучению.")

    # max_iter == 0 — агент сам решает: крутим до достижения требований, но не
    # больше AUTO_ATTEMPTS_CAP. max_iter > 0 — ровно столько попыток без раннего стопа.
    auto_mode = max_iter <= 0
    effective_max = AUTO_ATTEMPTS_CAP if auto_mode else max_iter
    total_display = f"авто (макс. {AUTO_ATTEMPTS_CAP})" if auto_mode else str(max_iter)

    for iter in range(1, effective_max + 1):
        iteration_context.set(iter)

        info_start_iter = f"Полный цикл обучения №{iter} из {total_display}"
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
            dataset_info=dataset_info,
            business_requirements=business_requirements,
            denied_hypotheses_info=denied_hypotheses_info,
            verbose=verbose,
            deployment_constraints=deployment_constraints,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            model_history=model_history
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
            model_id=model_id,
            ml_engin_out=ml_engin_out,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id
        )
        # Запуск процеса тренировки
        logger.info("")
        training_res = train_and_debug(training_input)

        info_final_iter_1_line = f"Полный цикл обучения №{iter} из {total_display}"
        info_final_iter_2_line =  "завершился с результатом:"
        if training_res.is_completed_successfully:
            logger.info(
                f"\n{'='*100}"
                f"\n{info_final_iter_1_line:^100}"
                f"\n{info_final_iter_2_line:^100}"
                f"\n{'✅ Модель успешно обучена':^100}"
                f"\n{'='*100}"
            )
            agent_history_client.info(f"Цикл обучения №{iter} из {total_display} завершён: модель успешно обучена.")

            # Аналитик метрик решает, достигнуты ли требования пользователя.
            # В авто-режиме (max_iter=0) достигнутые требования = досрочный стоп.
            # При фикс. количестве попыток цикл не прерываем (делаем ровно max_iter),
            # но разбор всё равно отдаём следующей попытке как контекст для улучшения.
            metrics_verdict = run_metrics_analyst(
                model_id=training_res.version_id,
                business_goal=business_requirements,
                verbose=verbose
            )
            if metrics_verdict.requirements_met:
                agent_history_client.info("Аналитик метрик: требования достигнуты.")
                if auto_mode:
                    logger.info("✅ Требования достигнуты — дальнейшие попытки не нужны.")
                    agent_history_client.info(
                        "Агент сам определил количество попыток — требования закрыты, "
                        "останавливаем цикл."
                    )
                    break
            else:
                agent_history_client.warning(
                    "Аналитик метрик: требования не достигнуты — нужен ещё этап обучения."
                )
                ml_model = ml_engin_out.ml_model
                what_trained = f"{ml_model.type} — {ml_model.description_model}" if ml_model else "модель"
                denied_hypotheses_info.append(
                    "Модель успешно обучилась, но требования пользователя не достигнуты.\n"
                    f"Что обучали: {what_trained}\n"
                    f"Вердикт аналитика: {metrics_verdict.reason}\n"
                    f"Разбор метрик:\n{metrics_verdict.analysis}\n"
                    "Учти это и предложи конфигурацию, которая закроет слабые места."
                )

        elif training_res.is_cancelled:
            # Человек вручную остановил обучение — это не сбой. Пайплайн не прерываем:
            # достаём частичные метрики отменённой версии, разбираем их и передаём
            # следующей итерации как контекст, чтобы агенты сообразили, почему человек
            # мог остановить, и предложили другую конфигурацию.
            logger.info(
                f"\n{'='*100}"
                f"\n{info_final_iter_1_line:^100}"
                f"\n{'обучение остановлено пользователем':^100}"
                f"\n{'='*100}"
            )
            agent_history_client.warning(
                f"Цикл обучения №{iter} из {total_display}: обучение остановлено пользователем. "
                "Разбираем частичные метрики, чтобы понять причину."
            )

            ml_model = ml_engin_out.ml_model
            what_trained = f"{ml_model.type} — {ml_model.description_model}" if ml_model else "модель"

            # Разбор частичных метрик отменённой версии (переиспользуем аналитика метрик).
            metrics_analysis = "Метрики на момент остановки недоступны."
            if training_res.version_id is not None:
                metrics_analysis = run_metrics_analyst(
                    model_id=training_res.version_id,
                    business_goal=business_requirements,
                    verbose=verbose
                ).to_history_text()

            denied_hypotheses_info.append(
                "Человек вручную остановил обучение этой конфигурации (это не ошибка обучения).\n"
                f"Что обучали: {what_trained}\n"
                f"Метрики и анализ на момент остановки:\n{metrics_analysis}\n"
                "Сделай вывод, почему человек мог остановить обучение, и предложи другую "
                "конфигурацию с учётом этого."
            )

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
                f"Цикл обучения №{iter} из {total_display} провалился: {training_res.error}"
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

