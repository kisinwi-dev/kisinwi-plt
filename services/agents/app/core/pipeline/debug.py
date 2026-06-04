from typing import Tuple
from .train import TrainingOut, TrainingInput, training
from app.core.crews.ml_debuger import run_ml_debug
from app.logs import get_logger

logger = get_logger(__name__)

def debugging(
    train_input: TrainingInput,
    training_res: TrainingOut,
    max_iter: int = 3
) -> Tuple[TrainingOut, int]:
    """
    Дебаг. Запускается процесс автоматического решения проблем обучения.
    Может решить проблемы связанные с настроенными конфигурациями.
    
    Args:
        train_input: Информация о предыдущем запуске
        training_res: Информация о результатах запуска

    Returns:
        TrainingOut: Результат аналогичный по структуре как результат обучения
        int: Версия модели
    """

    # автоматический отказ от дебагинга в случае, если всё обучилось
    if training_res.is_completed_successfully:
        return training_res, train_input.version_model

    logger.info("Проверка данных для включения дебаг режима...")
    # проверка входных данных
    if training_res.error is None:
        logger.error("❌ Нет ошибки для дебагинга")
        return TrainingOut(
            is_completed_successfully=False,
            error="Не удалось вызвать режим дебагинга: отсутствует ошибка"
        ), train_input.version_model

    # проверяем наличие конфигураций
    ml_model = train_input.ml_engin_out.ml_model
    if ml_model is None:
        logger.error("❌ Нет конфигурации модели для дебагинга")
        return TrainingOut(
            is_completed_successfully=False,
            error="Не удалось вызвать режим дебагинга: отсутствует конфигурация модели"
        ), train_input.version_model

    logger.info(
        f"\n{'='*50}"
        "\n ВКЛЮЧЕН ДЕБАГ РЕЖИМ"
        f"\n{'='*50}"
    )

    mod_version_model = 0

    # Попытки дебага
    for iteration in range(1, max_iter + 1):
        mod_version_model+=1
        logger.info(f"[ДЕБАГ] Итерация {iteration}/{max_iter}")
        logger.info("[ДЕБАГ] Агент-дебагер ищет решение...")
        # игнорируется None т.к. мы уверены, что там находятся переменные в формате str
        ml_debuger_out = run_ml_debug(
            error=training_res.error, # type: ignore[index]
            config=train_input.ml_engin_out.ml_model.configuration # type: ignore[index]
        )
        
        # Если решения не найдены
        if not ml_debuger_out.decision or ml_debuger_out.configuration is None:
            logger.error(f"[ДЕБАГ] ❌ Дебагер не нашёл решение: {ml_debuger_out.reason}")
            return TrainingOut(
                is_completed_successfully=False,
                error=f"Дебагер не может исправить: {training_res.error}"+
                    f"\n\nПояснения от дебагера: {ml_debuger_out.reason}"
            ), train_input.version_model
        
        logger.info("[ДЕБАГ] ✅ Агент-дебагер нашёл решение")        
        logger.info("[ДЕБАГ] Проверка найденного решения...")
        
        # Обновляем конфигурацию в ml_engin_out для следующей попытки
        train_input.ml_engin_out.ml_model.configuration = ml_debuger_out.configuration # type: ignore[index]
        train_input.version_model += mod_version_model 

        logger.info("[ДЕБАГ] Старт обучения с исправленной конфигурацией...")
        training_res = training(
            train_input
        )
        
        # Проверяем результат
        if training_res.is_completed_successfully:
            logger.info(f"[ДЕБАГ] ✅ Обучение успешно на итерации {iteration}!")
            return training_res, train_input.version_model
        else:
            logger.warning(f"[ДЕБАГ] ⚠️ Обучение снова не удалось: {training_res.error}")
    
    logger.error("[ДЕБАГ] ❌ Не удалось исправить ошибку после всех попыток")
    return TrainingOut(
        is_completed_successfully=False,
        error=f"Дебагер не помог после {max_iter} попыток. Последняя ошибка: {training_res.error}"
    ), train_input.version_model
