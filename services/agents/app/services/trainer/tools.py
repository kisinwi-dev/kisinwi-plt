import asyncio
from typing import Dict, Any
from crewai.tools import BaseTool

from ..utils import tool_response, get_json
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

TRAINER_URL = config_url.TRAINER['url']

class GetExampleTrainingConfigTool(BaseTool):
    """Инструмент для получения примера конфигурации тренировки модели"""

    name: str = "GetExampleTrainingConfig"
    description: str = """
    НАЗНАЧЕНИЕ: Получить пример конфигурации для запуска тренировки модели.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно понять структуру конфигурационного файла для обучения
    - В качестве шаблона перед созданием своей конфигурации
    - Когда не уверен в формате входных данных для тренировки

    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров

    ВОЗВРАЩАЕТ:
    - dict с информацией о настройке конфигурации обучения

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Используй как отправную точку для создания конфигурации
    - Не копируй слепо, адаптируй под свою задачу
    - Пример всегда актуален для текущей версии API
    """

    @tool_response(TRAINER_URL)
    def _run(self) -> str:
        url = f"{TRAINER_URL}/info/example_config"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self) -> str:
        return await asyncio.to_thread(self._run)


class GetAllAvailableModelsTool(BaseTool):
    """Инструмент для получения списка доступных архитектур ML моделей"""

    name: str = "GetAllAvailableModels"
    description: str = """
    НАЗНАЧЕНИЕ: Получить список всех доступных архитектур ML моделей с возможностью фильтрации.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать, какие модели доступны для обучения
    - Для поиска конкретной архитектуры (например, все resnet)
    - Перед выбором модели для своей задачи
    - Для изучения доступных опций нейросетевых архитектур

    ВХОДНЫЕ ДАННЫЕ:
    - filter_query (str): Фильтр для поиска моделей по названию.
      Поддерживает wildcard паттерны. ПЕРЕДАВАТЬ ФИЛЬТР В НИЖНЕМ РЕГИСТРЕ!!
      Примеры:
      - "*resnet*" - все модели, содержащие "resnet"
      - "*convnext*" - все модели ConvNeXt
      - "*efficient*" - все EfficientNet модели
      - "*" - все доступные модели

    ВОЗВРАЩАЕТ:
    - список моделей

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Известно, что используется библиотека timm 
    - Используй фильтр для сужения поиска, чтобы не перегружать ответ
    - Если filter пустой или "*", вернутся все доступные модели
    """

    @tool_response(TRAINER_URL)
    def _run(self, filter_query: str = "*") -> str:
        url = f"{TRAINER_URL}/info/ml_models"
        return get_json(url, params={"filter": filter_query}) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, filter_query: str = "*") -> str:
        return await asyncio.to_thread(self._run, filter_query)


class GetDeviceInfoTool(BaseTool):
    """Инструмент для получения информации об оборудовании"""

    name: str = "GetDeviceInfo"
    description: str = """
    НАЗНАЧЕНИЕ: Получить информацию о технических возможностях оборудования для обучения.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Перед началом обучения, чтобы понять доступные ресурсы
    - Для выбора оптимального batch_size и модели
    - Когда нужно узнать, доступна ли GPU
    - Для планирования времени обучения
    - При выборе между несколькими конфигурациями обучения

    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров

    ВОЗВРАЩАЕТ:
    - dict с информацией об оборудовании

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Всегда проверяй доступное устройство перед обучением
    - На CPU обучение будет значительно медленнее
    - Подбирай batch_size исходя из доступной памяти
    """

    @tool_response(TRAINER_URL)
    def _run(self) -> str:
        url = f"{TRAINER_URL}/info/device"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self) -> str:
        return await asyncio.to_thread(self._run)


class GetOptimizersTool(BaseTool):
    """Инструмент для получения списка доступных оптимизаторов"""

    name: str = "GetOptimizers"
    description: str = """
    НАЗНАЧЕНИЕ: Получить список всех доступных оптимизаторов для обучения.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - При выборе оптимизатора для обучения модели
    - Для изучения доступных оптимизаторов

    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров

    ВОЗВРАЩАЕТ:
    - список оптимизаторов

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Все оптимизаторы совместимы с PyTorch и используют ту же конфигурацию
    - Параметры оптимизатора передаются так же, как в оптимизаторы из torch.optim
    - Названия оптимизаторов чувствительны к регистру
    """

    @tool_response(TRAINER_URL)
    def _run(self) -> str:
        url = f"{TRAINER_URL}/info/optimizers"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self) -> str:
        return await asyncio.to_thread(self._run)


class GetSchedulersTool(BaseTool):
    """Инструмент для получения списка доступных планировщиков learning rate"""

    name: str = "GetSchedulers"
    description: str = """
    НАЗНАЧЕНИЕ: Получить список всех доступных планировщиков learning rate.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Для настройки динамического изменения learning rate во время обучения
    - Когда нужно улучшить сходимость модели
    - Для изучения доступных стратегий планирования
    - При выборе между разными планировщиками

    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров

    ВОЗВРАЩАЕТ:
    - список планировщиков

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Все планировщики совместимы с PyTorch и используют ту же конфигурацию
    - Параметры передаются так же, как в планировщики из torch.optim.lr_scheduler
    - Названия планировщиков чувствительны к регистру
    """

    @tool_response(TRAINER_URL)
    def _run(self) -> str:
        url = f"{TRAINER_URL}/info/schedulers"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self) -> str:
        return await asyncio.to_thread(self._run)


class GetMetricsForTrainerTool(BaseTool):
    """Инструмент для получения списка доступных метрик"""

    name: str = "GetMetricsForTrainer"
    description: str = """
    НАЗНАЧЕНИЕ: Получить список всех доступных метрик для оценки качества модели.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - При выборе метрик для оценки качества модели
    - Для понимания доступных метрик в системе
    - Перед настройкой конфигурации обучения
    - Когда нужно выбрать метрику для оптимизации

    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров

    ВОЗВРАЩАЕТ:
    - список доступных метрик

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Метрики чувствительны к регистру (используйте точно как в ответе)
    - loss всегда доступна во время обучения и добавляется автоматически(её не нужно указывать)
    - Рекомендуется использовать несколько метрик для комплексной оценки
    """

    @tool_response(TRAINER_URL)
    def _run(self) -> str: 
        url = f"{TRAINER_URL}/info/metrics"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self) -> str:
        return await asyncio.to_thread(self._run)