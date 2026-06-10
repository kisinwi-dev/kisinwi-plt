from .debug import debugging
from .train import training, TrainingInput, TrainingOut
from .reasoning import reasoning

def train_and_debug(
    training_input: TrainingInput
) -> TrainingOut:
    """
    Обучение модели. В случае ошибки в процессе обучения включается функция дебагинга.
    Режим дебагинга решает пытается решить проблему, если проблема связана с конфигурацией
    обучения.

    Args:
        training_input: Модель с требуемыми параметрами для запуска обучения

    Returns:
        TrainingOut: Информация об итогах обучения
    """
    training_res = training(training_input)
    training_res = debugging(training_input, training_res)
    return training_res
