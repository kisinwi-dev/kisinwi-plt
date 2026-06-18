import json
from typing import Optional, Type, List

from pydantic import BaseModel, ValidationError


class AgentOutput(BaseModel):
    """
    Единый контракт ответа агента.

    Помимо структурных полей (их переиспользуют другие агенты и пайплайн),
    каждый ответ умеет отдавать человекочитаемый markdown-нарратив «что сделал
    агент и почему». Этот нарратив идёт и в историю агентов, и в передачу между
    агентами — единое представление вместо разрозненных get_*-методов.
    """

    def to_history_text(self) -> str:
        """Markdown-нарратив ответа агента. Переопределяется в наследниках."""
        raise NotImplementedError


def extract_raw_text(crew_output) -> str:
    """Сырой текстовый вывод crew: raw первой задачи или строковое представление."""
    if crew_output is None:
        return ""
    if hasattr(crew_output, "raw"):
        return crew_output.raw
    if hasattr(crew_output, "tasks_output"):
        return crew_output.tasks_output[0].raw
    return str(crew_output)


def extract_json_objects(text: str) -> List[str]:
    """
    Top-level сбалансированные {...} блоки из текста.

    Агент при форс-финале (max_iter reached) клеит в ответ вывод инструментов
    перед настоящим JSON-объектом — нельзя парсить весь текст одним куском.
    Сканируем посимвольно с учётом строк и экранирования.
    """
    objects: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start:i + 1])
                    start = -1

    return objects


def parse_agent_output(raw: str, model: Type[BaseModel]) -> Optional[BaseModel]:
    """
    Валидирует raw в model, устойчиво к «грязному» выводу агента.

    Настоящий ответ агента — последний top-level JSON-объект, поэтому пробуем
    блоки с конца. None, если ни один не валиден (caller делает свой fallback).
    """
    if not raw:
        return None
    for block in reversed(extract_json_objects(raw)):
        try:
            return model.model_validate_json(block)
        except ValidationError:
            continue
    return None


def output_format_hint(model: Type[BaseModel]) -> str:
    """
    Подсказка формата ответа для промпта (JSON Schema модели).

    Заменяет авто-инъекцию схемы, которую раньше давал output_pydantic.
    """
    schema = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2)
    return (
        "Return ONLY a single valid JSON object matching this JSON Schema "
        "(no extra text, no tool output, no markdown fences):\n"
        f"{schema}"
    )


def _demo() -> None:
    """Self-check: грязный raw -> берём последний валидный объект."""
    class M(BaseModel):
        decision: bool
        reason: str

    dirty = (
        '{"configuration":"x"}{}{"validate": true, "errors": []}'
        '{"decision": true, "reason": "ok"}'
    )
    parsed = parse_agent_output(dirty, M)
    assert parsed is not None and parsed.decision is True and parsed.reason == "ok"

    clean = '{"decision": false, "reason": "no"}'
    assert parse_agent_output(clean, M).decision is False

    # строка с фигурными скобками внутри значения не ломает баланс
    nested = '{"decision": true, "reason": "a } b { c"}'
    assert parse_agent_output(nested, M).reason == "a } b { c"

    assert parse_agent_output("no json here", M) is None
    assert parse_agent_output("", M) is None

    # пустых top-level объектов недостаточно -> None (не падаем)
    assert parse_agent_output("{} {}", M) is None

    print("agent_output self-check OK")


if __name__ == "__main__":
    _demo()
