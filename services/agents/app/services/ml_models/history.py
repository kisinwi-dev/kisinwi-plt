import json

from .client import ml_models_client

NO_MODEL_HISTORY = (
    "Это новая модель — предыдущих версий и истории обучения нет. "
    "Составь конфигурацию с нуля."
)


def _truncate(text: str, max_chars: int) -> str:
    """Усечение длинного текста с пометкой."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…[усечено]"


def _as_text(value) -> str:
    """Привести значение (dict/list/str/None) к читаемому тексту."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def build_model_history_context(
    model: dict,
    max_versions: int = 8,
    max_detailed: int = 2,
    max_config_chars: int = 3000,
    max_metrics_chars: int = 600,
) -> str:
    """
    Собрать текстовый контекст истории версий существующей модели для промптов
    агентов: сводка по последним версиям + полные конфиги последних попыток.

    Конфиг обучения динамический — train_params не парсим, отдаём JSON как есть.
    """
    versions = model.get("versions") or []
    name = model.get("name", "—")
    description = model.get("description") or "—"

    lines = [
        f"Модель: «{name}»",
        f"Описание: {description}",
        f"Всего версий: {len(versions)}",
        "",
        "Ты обучаешь НОВУЮ ВЕРСИЮ этой существующей модели. Учитывай прошлые "
        "конфигурации и метрики: не повторяй решения провальных версий, "
        "отталкивайся от лучшей версии и целенаправленно улучшай её.",
    ]

    if not versions:
        lines.append("")
        lines.append("Версий ещё нет — это будет первая версия модели.")
        return "\n".join(lines)

    # Версии приходят от свежих к старым; сводка по последним max_versions.
    recent = versions[:max_versions]

    lines.append("")
    lines.append(f"## Последние версии (до {max_versions})")
    for v in recent:
        metrics = _truncate(_as_text(v.get("metrics_report")), max_metrics_chars)
        lines.append(
            f"- v{v.get('version')} | тип: {v.get('model_type', '—')} | "
            f"статус: {v.get('status', '—')} | метрики: {metrics}"
        )

    lines.append("")
    lines.append(f"## Конфигурации последних {max_detailed} версий")
    for v in recent[:max_detailed]:
        config = _truncate(_as_text(v.get("train_params")), max_config_chars)
        lines.append(f"### v{v.get('version')}")
        lines.append("```json")
        lines.append(config)
        lines.append("```")

    return "\n".join(lines)


def load_model_history(model_id: str | None) -> str | None:
    """
    История версий модели для контекста агентов.

    Returns:
        NO_MODEL_HISTORY — model_id не задан (новая модель);
        собранный контекст — модель найдена;
        None — model_id задан, но модель не найдена в реестре (caller должен
        остановить пайплайн).

    Логирование и системные сообщения остаются на стороне caller.
    """
    if model_id is None:
        return NO_MODEL_HISTORY
    model = ml_models_client.get_model(model_id)
    if model is None:
        return None
    return build_model_history_context(model)
