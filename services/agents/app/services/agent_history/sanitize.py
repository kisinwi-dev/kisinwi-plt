import re
from typing import Any

from app.core.memory import id_alias_context

# UUID любой версии в каноническом 8-4-4-4-12 виде.
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
)

# Маска для UUID, которым не нашлось читаемого имени.
_MASK = "[скрытый идентификатор]"


def _scrub_str(text: str) -> str:
    """Заменить UUID в строке: известные → имя, прочие → маска."""
    aliases = id_alias_context.get()

    def _replace(match: re.Match) -> str:
        uuid = match.group(0)
        return aliases.get(uuid, _MASK)

    return _UUID_RE.sub(_replace, text)


def scrub(value: Any) -> Any:
    """
    Рекурсивно очистить значение от UUID перед отправкой в историю агентов.

    UUID датасета/версии/модели текущего прогона заменяются на читаемое имя
    (из id_alias_context), любой другой UUID-подобный токен — на нейтральную
    маску. Структуру (dict/list/tuple) обходим рекурсивно, не-строковые
    скаляры возвращаем без изменений.
    """
    if isinstance(value, str):
        return _scrub_str(value)
    if isinstance(value, dict):
        return {key: scrub(val) for key, val in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item) for item in value)
    return value
