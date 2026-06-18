# Benchmark-прогон платформы

Прогон benchmark-датасетов с известным baseline через полный пайплайн воркфлоу. Бенчмарк только
**скачивает данные, запускает обучение и отдаёт ссылки на итог** (на дискуссию
агентов и обученную модель). Интерпретация результатов (метрики, кривые, сравнение
моделей) мы производим вручную во фронтенде по выданным ссылкам.

Поддерживаются два источника датасетов (поле `source` в `bench/config.py`):

- **huggingface** - публичные HF-датасеты, конвертируются `bench.prepare.hf`.
- **kaggle** - `master_kaggle`: пул из 6 Kaggle real-vs-deepfake датасетов, собирается
  `bench.prepare.kaggle`, сравнение с [референс-ноутбуком](https://www.kaggle.com/code/muqaddasejaz/deepfake-image-detection).

## Требования

Нужны только `uv` и запущенная платформа (`docker compose up --build` из корня).
Отдельная установка зависимостей не нужна: `uv run` сам поднимает окружение из
`pyproject.toml` при первом запуске.

Токены - в `.env` рядом со скриптом (нужен только под используемый источник):

```bash
cp .env.example .env   # HF_TOKEN и/или KAGGLE_API_TOKEN
```

## Единая точка входа

```bash
cd scripts/benchmarks

# весь HF-набор: подготовить недостающее + прогнать всё
uv run benchmark.py all
```

Базовый запуск в терминале спросит LLM-модель агентов (Enter - дефолт платформы).
Можно явно задать модель `--model <id>`.

### Варианты

```bash
# один датасет
uv run benchmark.py beans

# несколько конкретных
uv run benchmark.py cifar10 flowers102

# конкретная LLM-модель агентов
uv run benchmark.py all --model openai/gpt-5.1

# только прогнать (датасеты уже в платформе)
uv run benchmark.py all --skip-prepare
```

Адреса сервисов берутся из `bench/config.py` (дефолт - localhost) и переопределяются env:
`BENCH_DATASETS_URL`, `BENCH_AGENTS_URL`, `BENCH_TASKER_URL`,
`BENCH_AGENT_HISTORY_URL`, `BENCH_FRONTEND_URL` (последний - база ссылок на
дискуссию и модель в результатах).

## Kaggle deepfake-пул

`master_kaggle` не входит в `all` - Kaggle-данные тяжёлые (~десятки ГБ). Запускается
по имени, как и остальные: сырьё качается автоматически, если его ещё нет
(нужен `KAGGLE_API_TOKEN`).

```bash
# Собрать единый датасет + прогнать (сырьё подтянется само)
uv run benchmark.py master_kaggle
```

Можно скачать сырьё заранее отдельной командой (например, на быстром канале):

```bash
uv run benchmark.py download            # оставляет .zip
uv run benchmark.py download --unzip    # сразу распаковать
```

Раскладка повторяет ноутбук: стратифицированный split 80/10/10 (seed=42),
классы `Real`/`Fake`, метки по структуре папок. Файлы - hardlink, zip +20 ГБ
грузится потоково (`bench.common.upload_archive`), иначе `requests` падает с MemoryError.

Self-check логики split: `uv run python -c "from bench.prepare.kaggle import demo; demo()"`.

## Результат

Итог прогона пишется в `data/results.json` (мерж по key - можно прогонять порциями):
ссылки на дискуссию и обученную модель, статусы, число задач обучения, `baseline`
для сравнения. Метрики не дублируются - источник правды сервисы и UI платформы.

## Датасеты (bench/config.py)

| key | источник | датасет | классы | baseline |
|---|---|---|---|---|
| beans | hf | Beans | 3 | ~98% |
| cifar10 | hf | CIFAR-10 | 10 | ~96% |
| oxford_pets | hf | Oxford-IIIT Pets | 37 | ~93% |
| flowers102 | hf | Oxford Flowers-102 | 102 | ~96% |
| food101 | hf | Food-101 (subset 100/класс) | 101 | ~80% |
| master_kaggle | kaggle | Deepfake Real vs Fake (пул 6 Kaggle) | 2 | ~99% |

baseline (hf) - публичный accuracy на test данных с ResNet50.
baseline master_kaggle - точность референс-ноутбука на test (~99.3%).
