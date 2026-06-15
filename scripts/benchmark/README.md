# Benchmark-прогон платформы KiSinWi

Прогон benchmark-датасетов с известным baseline через полный пайплайн
(datasets → agents → tasker → trainer → metrics). Бенчмарк делает только суть:
**скачивает данные, запускает обучение и отдаёт ссылки на итог** — дискуссию
агентов и обученную модель. Интерпретация результатов (метрики, кривые, сравнение
моделей) — во фронтенде по выданным ссылкам.

## Требования

Нужны только `uv` и запущенная платформа (`docker compose up --build` из корня,
сервисы healthy). Отдельная установка зависимостей не нужна: `uv run` сам поднимает
окружение из `pyproject.toml` при первом запуске.

Для скачивания датасетов с HuggingFace положи `.env` рядом со скриптом с токеном
(см. `.env.example`):

```bash
cp .env.example .env   # и впиши HF_TOKEN
```

## Единая точка входа

```bash
cd scripts/benchmark

# весь набор: подготовить недостающее + прогнать всё
uv run benchmark.py all
```

Базовый запуск в терминале интерактивно спросит LLM-модель агентов (Enter — дефолт
платформы). В неинтерактивном запуске (CI, пайп) берётся дефолт без вопросов; явно
модель задаёт `--model <id>`.

### Варианты

```bash
# один датасет
uv run benchmark.py beans

# несколько конкретных
uv run benchmark.py cifar10 flowers102

# конкретная LLM-модель агентов (без интерактива)
uv run benchmark.py all --model openai/gpt-5.1

# только прогнать (датасеты уже загружены в платформу)
uv run benchmark.py all --skip-prepare
```

Опции: `--max-iter N` (итераций пайплайна, по умолчанию `DEFAULT_MAX_ITER` из
config.py = 3 — даёт платформе восстановиться после неудачного выбора архитектуры),
`--skip-prepare`, `--model <id>`, `--out FILE`.

Адреса сервисов берутся из config.py (дефолт — localhost) и переопределяются env:
`BENCH_DATASETS_URL`, `BENCH_AGENTS_URL`, `BENCH_TASKER_URL`,
`BENCH_AGENT_HISTORY_URL`, `BENCH_FRONTEND_URL` (последний — база ссылок на
дискуссию и модель в результатах; напр. для docker-hostname или удалённого стенда).

## Результат

Итог прогона пишется в `results.json` (мерж по key — можно прогонять порциями).
Для каждого датасета — ссылки на итог обучения:

- `discussion_id` / `discussion_url` — дискуссия агентов (`/agents/discussion/<id>`)
- `model_id` / `model_url` — обученная модель (`/models/<id>`)
- статусы прогона (`task_status`, `discussion_status`, `task_error`),
  `n_train_tasks` (сколько раз обучали — recovery при `max_iter`), `duration_sec`.

Метрики не дублируются: источник правды — сервисы и UI платформы, смотрим их по
выданным ссылкам.

## Датасеты (config.py)

| key | датасет | классы | baseline (ResNet/transfer) |
|---|---|---|---|
| beans | Beans | 3 | ~98% |
| cifar10 | CIFAR-10 | 10 | ~96% |
| oxford_pets | Oxford-IIIT Pets | 37 | ~93% |
| flowers102 | Oxford Flowers-102 | 102 | ~96% |
| food101 | Food-101 (subset 100/класс) | 101 | ~80% |

`all` = `DEFAULT_SUITE` из config.py. `baseline` уходит в `business_requirements`
прогона как целевая точность. Новый датасет добавляется одной записью в
`config.DATASETS` (hf_id, колонки, split_map, baseline).

## Скрипты (можно вызывать по отдельности)

- `benchmark.py` — оркестратор (prepare + run).
- `prepare_dataset.py <key>` — скачать с HF, конвертировать в train/val/test,
  загрузить в datasets. Пишет `datasets_registry.json` (key → UUID).
- `run_benchmark.py <keys...> [--max-iter N]` — прогон пайплайна, ожидание
  завершения всего workflow и запись ссылок на итог в `results.json`.
- `config.py` / `common.py` — конфигурация и HTTP-хелперы.
