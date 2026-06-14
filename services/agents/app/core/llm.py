import time
from dataclasses import dataclass
from typing import List, Optional

import requests
from crewai import LLM

from app.config import config_base_llm
from app.core.memory import llm_model_context
from app.logs import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """
    Описание модели в каталоге.

    id                  — id модели OpenRouter (то, что уходит в LLM(model=...)).
    label               — человекочитаемое имя для UI.
    supports_temperature — принимает ли модель кастомный temperature.
                           Часть моделей (напр. gpt-5*) допускает только дефолт
                           провайдера — для них temperature не передаём.
    notes               — заметки/ограничения.
    """
    id: str
    label: str
    supports_temperature: bool = True
    notes: str = ""


# Каталог моделей сервис НЕ хранит у себя — он тянется из OpenRouter
# (OpenAI-совместимый эндпоинт {OPENAI_API_BASE}/models). Ответ кэшируется на
# короткое время, чтобы не дёргать провайдера на каждый запрос настроек.
_CATALOG_TTL_SECONDS = 300

# Допускаем только модели OpenAI. Наши tool- и output_pydantic-схемы заточены
# под OpenAI JSON Schema (напр. Anthropic-провайдеры отвергают minimum/maximum
# у integer-полей). Префикс — как в id OpenRouter: "openai/...".
_ALLOWED_PROVIDER_PREFIX = "openai/"
_catalog_cache: List[ModelInfo] = []
_catalog_fetched_at: float = 0.0


def _models_endpoint() -> str:
    base = (config_base_llm.OPENAI_API_BASE or "").rstrip("/")
    return f"{base}/models"


def _is_crew_compatible(supported: object) -> bool:
    """
    Совместима ли модель с нашим пайплайном по заявленным параметрам.

    Минимизируем ошибки в рантайме: агенты делают и function-calling, и строгий
    structured output (output_pydantic). Поэтому требуем одновременно:
      * tools + tool_choice          — надёжный вызов инструментов;
      * structured_outputs ИЛИ response_format — способность к строгому JSON.

    Если провайдер не сообщил supported_parameters списком — модель НЕ
    пропускаем (лучше скрыть сомнительную модель, чем сломать пайплайн).
    Это фильтр по возможностям, а не свой список конкретных моделей.
    """
    if not isinstance(supported, list):
        return False
    if "tools" not in supported or "tool_choice" not in supported:
        return False
    return "structured_outputs" in supported or "response_format" in supported


def _parse_models_response(data: List[dict]) -> List[ModelInfo]:
    """Разобрать ответ OpenRouter /models в каталог совместимых моделей."""
    catalog: List[ModelInfo] = []
    for item in data:
        model_id = item.get("id")
        if not model_id:
            continue
        if not model_id.startswith(_ALLOWED_PROVIDER_PREFIX):
            continue
        supported = item.get("supported_parameters")
        if not _is_crew_compatible(supported):
            continue
        catalog.append(ModelInfo(
            id=model_id,
            label=item.get("name") or model_id,
            supports_temperature="temperature" in supported,
        ))
    catalog.sort(key=lambda m: m.id)
    return catalog


def _fetch_catalog() -> List[ModelInfo]:
    headers = {}
    if config_base_llm.OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {config_base_llm.OPENROUTER_API_KEY}"
    resp = requests.get(_models_endpoint(), headers=headers, timeout=15)
    resp.raise_for_status()
    return _parse_models_response(resp.json().get("data", []))


def get_catalog(force_refresh: bool = False) -> List[ModelInfo]:
    """
    Каталог доступных моделей из OpenRouter (с кэшем).

    При сбое запроса отдаём последний успешный кэш (возможно пустой) — настройки
    и запуск пайплайна не должны падать из-за недоступности каталога: выбранную
    модель всё равно можно указать вручную.
    """
    global _catalog_cache, _catalog_fetched_at
    now = time.time()
    if not force_refresh and _catalog_cache and now - _catalog_fetched_at < _CATALOG_TTL_SECONDS:
        return _catalog_cache
    try:
        _catalog_cache = _fetch_catalog()
        _catalog_fetched_at = now
    except Exception as e:
        logger.error(f"Не удалось получить каталог моделей из OpenRouter: {e}")
    return _catalog_cache


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Найти модель в каталоге по id. None — если модели нет (кастомная)."""
    return next((m for m in get_catalog() if m.id == model_id), None)


# Дефолтная модель — из env (OPENAI_MODEL_NAME). Рантайм-состояние выбранной
# модели держим на уровне модуля (один процесс сервиса).
DEFAULT_MODEL_ID: str = config_base_llm.OPENAI_MODEL_NAME
_current_model_id: str = DEFAULT_MODEL_ID


def get_current_model() -> str:
    """Глобально выбранная модель (без учёта per-request override)."""
    return _current_model_id


def set_current_model(model_id: str) -> None:
    """Сменить глобально выбранную модель агентов в рантайме."""
    global _current_model_id
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("model_id не может быть пустым")
    if get_model_info(model_id) is None:
        logger.warning(f"Модель '{model_id}' отсутствует в каталоге — используется как кастомная.")
    _current_model_id = model_id
    logger.info(f"Модель агентов изменена на '{model_id}'")


def resolve_model_id() -> str:
    """
    Какая модель используется прямо сейчас.

    Приоритет: per-request override (contextvar) → глобальная настройка → env-дефолт.
    """
    return llm_model_context.get() or _current_model_id


def model_supports_temperature(model_id: str) -> bool:
    """
    Поддерживает ли модель кастомный temperature.

    Для модели из каталога — её флаг; для кастомной — fallback на глобальный
    env-флаг LLM_TEMPERATURE_SUPPORTED.
    """
    info = get_model_info(model_id)
    if info is not None:
        return info.supports_temperature
    return config_base_llm.LLM_TEMPERATURE_SUPPORTED


def to_openrouter_model(model_id: str) -> str:
    """
    Принудительно маршрутизировать модель через OpenRouter (OpenAI-совместимый
    провайдер CrewAI), добавив префикс `openrouter/`.

    Без префикса CrewAI 1.x по префиксу провайдера (`anthropic/`, `google/`, ...)
    пытается поднять нативный SDK провайдера (для Anthropic это `crewai[anthropic]`,
    которого у нас нет) и/или обрезает `org/` из id. С префиксом `openrouter/`
    запрос всегда идёт через OpenAI-совместимый клиент на OPENAI_API_BASE с
    OPENROUTER_API_KEY, а полный `org/model` сохраняется — так работают все LLM.
    """
    model_id = model_id.strip()
    return model_id if model_id.startswith("openrouter/") else f"openrouter/{model_id}"


def make_llm(temperature: float) -> LLM:
    """
    Создаёт LLM-инстанс с заданным temperature для текущей выбранной модели.

    Модель резолвится на момент вызова (resolve_model_id), поэтому смена модели
    в рантайме или per-request override применяются при построении crew.
    Если модель не поддерживает кастомный temperature — параметр не передаётся
    и используется дефолт провайдера.
    """
    model_id = resolve_model_id()
    kwargs = dict(
        model=to_openrouter_model(model_id),
        base_url=config_base_llm.OPENAI_API_BASE,
        api_key=config_base_llm.OPENROUTER_API_KEY,
    )
    if model_supports_temperature(model_id):
        kwargs["temperature"] = temperature
    return LLM(**kwargs)


def get_llm_precise() -> LLM:
    """
    Низкий temperature — детерминированный JSON/факты:
    ml_engineer, ml_engineer_quick, ml_debuger, dataset_analyst,
    metrics_analyst, reporter, searcher'ы.
    """
    return make_llm(temperature=0.15)


def get_llm_creative() -> LLM:
    """Выше temperature — генерация гипотез (researcher)."""
    return make_llm(temperature=0.7)
