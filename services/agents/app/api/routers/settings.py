from fastapi import APIRouter

from app.api.schemas import LlmModelInfo, LlmSettingsResponse, SetLlmModelRequest
from app.core.llm import (
    DEFAULT_MODEL_ID,
    get_catalog,
    get_current_model,
    get_model_info,
    set_current_model,
)

routers = APIRouter(
    prefix="/settings",
    tags=["settings"],
)


def _build_settings_response() -> LlmSettingsResponse:
    current = get_current_model()
    return LlmSettingsResponse(
        current_model=current,
        default_model=DEFAULT_MODEL_ID,
        is_custom=get_model_info(current) is None,
        available=[
            LlmModelInfo(
                id=m.id,
                label=m.label,
                supports_temperature=m.supports_temperature,
                notes=m.notes,
            )
            for m in get_catalog()
        ],
    )


@routers.get(
    "/llm",
    summary="Текущая модель агентов и каталог",
    description="Возвращает выбранную модель агентов, дефолт и каталог доступных моделей",
    response_model=LlmSettingsResponse,
)
def get_llm_settings() -> LlmSettingsResponse:
    return _build_settings_response()


@routers.put(
    "/llm",
    summary="Сменить модель агентов",
    description="Меняет глобально выбранную модель агентов в рантайме (без перезапуска). "
                "Модель вне каталога допускается, но помечается как кастомная.",
    response_model=LlmSettingsResponse,
)
def set_llm_settings(req: SetLlmModelRequest) -> LlmSettingsResponse:
    set_current_model(req.model)
    return _build_settings_response()
