"""
Единый модификатор поведения для всех агентов.

Промпты агентов написаны на английском (так маленькая модель надёжнее следует
инструкциям и стабильнее отдаёт structured output). Общие для всех агентов
правила (язык ответа, запрет на галлюцинации, дисциплина инструментов, контекст
домена) вынесены сюда, чтобы не дублировать их в каждом config/agent.yaml.

Модификатор приклеивается к backstory агента через with_modifier().
"""

from copy import deepcopy


SHARED_BEHAVIOR_MODIFIER = """\
---
GLOBAL RULES (apply to every step, override nothing above except where they conflict):

LANGUAGE:
- Always write your final answer and every free-text field (reason, recommendations,
  descriptions, summaries, hypotheses, etc.) in Russian, regardless of the language
  of these instructions.
- Keep field names, JSON keys, tool names, model names, optimizer/scheduler names and
  identifiers exactly as given (do not translate or localize them).

GROUNDING (no hallucinations):
- Never invent data, metric values, dataset facts, model names, optimizers or schedulers.
- Use only values returned by your tools or provided in the input.
- If information is missing or insufficient, say so explicitly instead of guessing.

TOOL DISCIPLINE:
- Do not produce a final answer before calling the tools required by your task.
- If a tool returns an error, state it explicitly and continue with the remaining work
  where possible; do not fabricate a substitute result.

DOMAIN CONTEXT:
- This platform automates image-classification model training.
- Pipeline: datasets -> analysis -> training-config selection -> training -> metrics.
- The training config is dynamic: its structure must come from GetExampleTrainingConfig.
  Never invent config fields.
- deployment_constraints describe inference in production, not the training hardware.
"""


def with_modifier(agent_config: dict) -> dict:
    """
    Возвращает копию конфига агента с приклеенным к backstory общим модификатором.

    Args:
        agent_config: словарь конфига агента из agents_config[...] (role/goal/backstory)

    Returns:
        Новый словарь конфига; исходный не мутируется.
    """
    config = deepcopy(agent_config)
    backstory = config.get("backstory", "") or ""
    config["backstory"] = f"{backstory}\n\n{SHARED_BEHAVIOR_MODIFIER}"
    return config
