# metrics

Сервис хранения и выдачи метрик платформы KiSinWi.
**FastAPI + MongoDB** (pymongo), порт **6310**, БД `metrics_mongo` (порт **6315**).
Зависимости - через **uv** (`uv sync`, `uv run ...`).

> Это документация для разработчика: назначение, запуск, конфиг, HTTP API, онлайн-поток
> (SSE), модель данных, логика сравнения и взаимодействие с сервисами. Карта модулей `app/` -
> в конце файла.

## Назначение и место в пайплайне

metrics хранит две независимые области данных:

```
datasets → agents → tasker → trainer → [ metrics ] + ml_models → agent_history
```

- **Метрики моделей** - значения по эпохам (loss, accuracy, ...), разбитые по выборкам
  train/val/test. Их пишет **trainer** раз в эпоху; поверх сырых значений сервис отдаёт
  сводную статистику, сравнение моделей и онлайн-поток (SSE).
- **Метрики агентов** - использование LLM (токены, стоимость и т.п.) по каждому ответу
  агента; агрегируются по дискуссии.

Сервис сам никуда не ходит: все вызовы входящие. Пишет в него **trainer**, читают -
**agents / frontend**.

## Архитектура

`main.py` поднимает один uvicorn-сервер. Две области обслуживают два менеджера-синглтона
([app/api/deps.py](app/api/deps.py)), общих для всех запросов:

- `CVMetricManager` ([app/core/model.py](app/core/model.py)) - метрики моделей (коллекция
  `training_cv`);
- `AgentsResponseManager` ([app/core/agent.py](app/core/agent.py)) - метрики агентов
  (коллекция `agent_response`).

Оба наследуют `ManagerBase` ([app/core/mongo.py](app/core/mongo.py)): держат собственный
`MongoClient`, который открывается в `lifespan` (`connect`) и закрывается на остановке
(`disconnect`). Пула на запрос нет - клиент pymongo потокобезопасен и переиспользуется.

Статистика вынесена в чистые функции над схемами ([app/core/stats.py](app/core/stats.py)):
`compute_model_summary`, `compare_models`. Онлайн-поток обслуживает in-memory брокер
`MetricStreamBroker` ([app/core/stream.py](app/core/stream.py)) - тоже синглтон в deps.

```
                  ┌──────────────── metrics (main.py, uvicorn) ────────────────┐
                  │                                                             │
  trainer ──────▶ │  POST /models/add|adds|{id}/status|checkpoint|class-report │
                  │        │            │                                       │
                  │        ▼            ▼ publish                               │
                  │  CVMetricManager   MetricStreamBroker ──▶ GET /models/{id}/stream (SSE)
                  │        │                                       ▲            │
  frontend ─────▶ │  GET /models/{id}, /summary, POST /compare ───┘            │
                  │        │                                                    │
                  │        ▼                                                    │
                  │  MongoDB training_cv │ agent_response  ◀── AgentsResponseManager
  agents ───────▶ │  POST /agents/add, GET /agents/discussions/{id}            │
                  └─────────────────────────────────────────────────────────────┘
```

## Запуск

```bash
uv sync
python main.py          # uvicorn на 0.0.0.0:6310, reload=True
# либо: uv run uvicorn main:app --host 0.0.0.0 --port 6310
```

В `lifespan` ([main.py:15-33](main.py#L15)) вызывается `check_health_all()`, но он **не
блокирует старт**: при недоступной MongoDB сервис всё равно поднимется, только индексы не
создадутся (`ensure_indexes` пропускается, см. [main.py:24-28](main.py#L24)), а запросы к БД
будут падать с 500. Порт в `main.py` захардкожен (`6310`); `METRICS_SERVICE_PORT` читается
только в Docker CMD.

### Docker

```dockerfile
FROM python:3.x-slim
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${METRICS_SERVICE_PORT:-6310}"]
```

## Переменные окружения

Читаются в [app/config.py](app/config.py) (`MongoDBConfig`).

| Переменная               | Назначение                          | По умолчанию       |
|--------------------------|-------------------------------------|--------------------|
| `MONGO_HOST`             | хост MongoDB                        | `localhost`        |
| `MONGO_PORT`             | порт MongoDB                        | `6315`             |
| `MONGO_APP_USERNAME`     | пользователь БД                     | `metrics_service`  |
| `MONGO_APP_PASSWORD`     | пароль БД                           | задан в коде       |
| `MONGO_METRIC_DATABASE`  | имя базы                            | `metrics`          |
| `METRICS_SERVICE_PORT`   | порт сервиса (только в Docker CMD)  | `6310`             |

URL собирается как `mongodb://{user}:{password}@{host}:{port}/{database}`
([config.py:14-16](app/config.py#L14)). В отличие от tasker, пустой пароль исключения не
бросает: подключение просто не установится, и это всплывёт на healthcheck.

## HTTP API

Роутеры: [app/api/routers/model.py](app/api/routers/model.py) (префикс `/models`),
[app/api/routers/agent.py](app/api/routers/agent.py) (префикс `/agents`),
[app/api/routers/info.py](app/api/routers/info.py) (префикс `/info`).

### Метрики моделей (`/models`)

| Метод  | Путь                       | Назначение |
|--------|----------------------------|------------|
| POST   | `/add`                     | добавить одну метрику; при отсутствии модели создаёт документ (upsert) |
| POST   | `/adds`                    | добавить пачку метрик за раз (так шлёт trainer раз в эпоху) |
| POST   | `/batch`                   | метрики нескольких моделей за один запрос (без метрик - не в ответе) |
| POST   | `/compare`                 | сравнить N моделей (минимум 2) по метрикам выборки |
| GET    | `/{model_id}`              | все метрики модели по выборкам; 404 если нет |
| GET    | `/{model_id}/stream`       | **SSE**: снимок метрик при подключении и после каждой записи + событие `end` |
| GET    | `/{model_id}/summary`      | сводка final/best/min/max + разрывы train/val; 404 если нет |
| GET    | `/{model_id}/exists`       | есть ли у модели сохранённые метрики (boolean) |
| POST   | `/{model_id}/status`       | статус обучения от trainer (in_progress / completed / failed / cancelled) |
| POST   | `/{model_id}/checkpoint`   | инфо о сохранённых весах `{epoch, metric, value}` (идемпотентно) |
| POST   | `/{model_id}/class-report` | confusion matrix + per-class P/R/F1 на test (идемпотентно) |
| GET    | `/{model_id}/class-report` | сохранённый отчёт по классам; 404 если нет |
| DELETE | `/{model_id}`              | удалить все метрики модели; 404 если нет |

### Метрики агентов (`/agents`)

| Метод  | Путь                            | Назначение |
|--------|---------------------------------|------------|
| POST   | `/add`                          | метрики одного ответа агента; **409** если уже существуют |
| GET    | `/discussions/{discussion_id}`  | метрики всех ответов дискуссии + суммарная сводка по токенам |
| GET    | `/{response_id}`                | метрики ответа по id; 404 если нет |
| DELETE | `/{response_id}`                | удалить метрики ответа; 404 если нет |

### Служебные (`/info`)

| Метод | Путь      | Назначение |
|-------|-----------|------------|
| GET   | `/health` | состояние сервиса и MongoDB |

## Онлайн-поток метрик (SSE)

`GET /models/{model_id}/stream` отдаёт `text/event-stream`
([model.py:140-216](app/api/routers/model.py#L140)):

- при подключении сразу шлётся событие `metrics` с полным снимком (схема `ModelMetrics`);
- затем такой же снимок шлётся после каждой записи (`/add`, `/adds`, `/status`,
  `/checkpoint`, удаление) - роутеры записи вызывают `broker.publish(model_id)`;
- когда trainer ставит финальный статус, после снимка идёт событие `end`
  (`{model_id, status}`) - **один раз** на соединение. Подписчику уже завершённой модели
  снимок и `end` приходят сразу;
- после `end` сервер поток **не закрывает** - иначе клиентский `EventSource` уйдёт в
  бесконечные переподключения; закрывает соединение сам клиент;
- каждые 15 секунд шлётся keepalive-комментарий (`: ping`), чтобы прокси не рвали
  соединение;
- подключаться можно **до первой эпохи**: для модели без метрик отдаётся пустой снимок.

**Подводный камень.** Брокер - `asyncio.Queue` в памяти процесса. Уведомления видны только
внутри того же uvicorn-процесса. Несколько воркеров (`--workers > 1`) или балансировщик
поверх нескольких инстансов **сломают доставку**: запись в одном процессе не разбудит
подписчика в другом. `class_report` в снимок **не входит** - фронт запрашивает его отдельно
после события `end`.

## Модель данных

Две коллекции в БД `metrics`. У обеих - уникальный индекс, создаётся в `ensure_indexes`
при старте (если БД доступна).

### `training_cv` - метрики моделей

Уникальный индекс: `model_id`. Документ:

```jsonc
{
  "model_id": "uuid",
  "status": "in_progress|completed|failed|cancelled",  // у старых моделей отсутствует
  "splits": {
    "train": [
      {
        "metric": "loss",                 // чистое имя, без префикса train_/val_/test_
        "values": [0.91, 0.55, 0.42],     // по эпохам
        "timestamps": ["2026-06-11T10:00:00Z", "..."]  // UTC, параллельны values
      }
    ],
    "val":  [ { "metric": "loss", "values": [...], "timestamps": [...] } ],
    "test": [ ... ]
  },
  "checkpoint": {                          // от trainer после обучения; у старых нет
    "epoch": 13,                           // эпоха сохранённых весов (нумерация с 1)
    "metric": "loss",                      // early-stop-метрика (чистое имя, val-выборка)
    "value": 0.42                          // value=null - веса финальной эпохи
  },
  "class_report": {                        // один раз на test; у старых нет
    "labels": ["cat", "dog"],
    "confusion_matrix": [[8, 2], [1, 9]],  // строки - истинные, столбцы - предсказанные
    "per_class": [ { "label": "cat", "precision": 0.89, "recall": 0.8, "f1": 0.84, "support": 10 } ]
  }
}
```

Неочевидное:

- **Разбиение по выборкам явное.** На записи выборка берётся из поля `split` метрики, при
  его отсутствии - из префикса названия `train_`/`val_`/`test_` (так шлёт trainer). Имя без
  известного префикса относится к `train` (`parse_split` в [app/core/model.py](app/core/model.py)).
  В хранилище и в ответах имена метрик уже **чистые**, без префиксов.
- **`timestamps` параллелен `values`.** При записи через `/adds` ставится одна метка на
  пачку; если в payload меток нет - подставляется серверное время. У старых документов массив
  может быть **короче** `values` - тогда выравнивание идёт с конца.
- `status`, `checkpoint`, `class_report` - **опциональны**: у старых моделей их нет
  (читаются как `null` / 404).
- Плоский формат до разбиения по выборкам мигрируется скриптом
  `uv run python -m scripts.migrate_to_splits`.

### `agent_response` - метрики агентов

Уникальный индекс: `response_id`. Документ:

```jsonc
{
  "response_id": "uuid",     // уникальный
  "discussion_id": "uuid",   // к какой дискуссии относится ответ
  "metrics": {               // свободный словарь чисел/флагов
    "input_tokens": 150,
    "output_tokens": 250,
    "total_tokens": 400,
    "cost": 0.005,
    "is_cached": false
  }
}
```

Сводка по дискуссии (`GET /agents/discussions/{id}`) суммирует **только числовые** поля
`metrics` по всем ответам; `bool` (например `is_cached`) в сумму не входит, плюс добавляется
`responses_count` ([agent.py:57-78](app/core/agent.py#L57)).

## Логика сравнения и статистики

Направление метрики - эвристика по названию ([stats.py:17-23](app/core/stats.py#L17)):
если в имени есть `loss`/`error`/`err`/`mae`/`mse`/`rmse` - меньше лучше, иначе больше лучше.

- **`/summary`** (`compute_model_summary`) - по каждой метрике каждой выборки даёт
  `final`/`best`/`min`/`max`, эпоху лучшего и число эпох; плюс разрывы train/val на последней
  эпохе (`gap > 0` - на валидации хуже, признак переобучения).
- **`/compare`** (`compare_models`) - лидер по метрике определяется по `weights_value`:
  значению на **эпохе сохранённых весов** (срез `values` по `checkpoint.epoch`). Если
  чекпоинта нет (или это одноточечный test, где значение уже измерено на сохранённых весах) -
  фолбэк на `final_value`. Модели без метрик попадают в `missing` (это не ошибка),
  `delta_best` - модуль отставания от лидера.

## Статусы обучения

`POST /models/{id}/status` шлёт trainer:

```
   in_progress ──▶ completed
       │      └──▶ failed
       └──────────▶ cancelled
```

`in_progress` на старте **сбрасывает** ранее выставленный финальный статус (переобучение той
же модели). Финальные статусы (`completed`/`failed`/`cancelled`) дальше отдаются подписчикам
SSE событием `end`. Если документа ещё нет - он создаётся (upsert).

## Взаимодействие с сервисами

Все вызовы входящие, исходящих нет.

| Источник | Что вызывает | Когда |
|----------|--------------|-------|
| trainer  | `POST /models/adds` | значения метрик раз в эпоху |
| trainer  | `POST /models/{id}/status` | старт и итог обучения |
| trainer  | `POST /models/{id}/checkpoint` | один раз после обучения |
| trainer  | `POST /models/{id}/class-report` | один раз после оценки на test |
| agents   | `POST /agents/add` | метрики ответа агента |
| frontend | `GET /models/{id}`, `/summary`, `/stream`, `POST /compare` | отображение метрик |

`GET /info/health` проверяет **только** MongoDB ([app/core/health.py](app/core/health.py)):
`HEALTHY` при успешном `ping`, иначе `DEGRADED`/`UNHEALTHY`.

## Нюансы

- **SSE-брокер живёт в одном процессе.** `MetricStreamBroker` - `asyncio.Queue` в памяти процесса;
  несколько воркеров uvicorn (`--workers > 1`) или несколько инстансов **сломают** доставку: запись
  в одном процессе не разбудит подписчика в другом.
- **`class_report` не входит в SSE-снимок** - фронт запрашивает его отдельно после события `end`.
- **`timestamps` параллелен `values`**, но у старых документов может быть короче - тогда
  выравнивание идёт **с конца**.
- **`status`/`checkpoint`/`class_report` опциональны**: у старых моделей их нет (читаются как
  `null`/404).
- **Направление метрики - эвристика по названию**: `loss`/`error`/`mae`/`mse`/`rmse` - меньше лучше,
  иначе больше лучше.
- **Лидер `/compare` - по `weights_value`** (значение на эпохе сохранённых весов), фолбэк на
  `final_value` у моделей без чекпоинта.
- **Пустой пароль исключения не бросает** (в отличие от tasker): подключение просто не установится
  и всплывёт на healthcheck.
- **Порт захардкожен** в `main.py` (`6310`); `METRICS_SERVICE_PORT` читается только в Docker CMD.

## Отладка

- **Логи**: `app/logs/app_json.log`, уровень `DEBUG`.
- **БД недоступна при старте** → сервис поднимается, но индексы не создаются; запросы к БД
  падают с 500, health отдаёт `DEGRADED`.
- **Несколько воркеров uvicorn** → SSE-уведомления не доходят до подписчиков из других
  процессов (in-memory брокер); запускать в один процесс.
- **`POST /agents/add` вернул 409** → метрики этого `response_id` уже записаны (защита
  уникальным индексом).
- **Старые документы** без `status`/`checkpoint`/`class_report` или с укороченным
  `timestamps` - норма; миграция формата: `uv run python -m scripts.migrate_to_splits`.

## Структура `app/`

```
app/
├── config.py          MongoDBConfig (env, URL подключения)
├── api/
│   ├── routers/       model.py (метрики моделей + SSE), agent.py (метрики агентов), info.py (health)
│   ├── schemas/       model.py, agent.py, info.py (Pydantic-схемы запросов/ответов)
│   └── deps.py        синглтоны: CVMetricManager, AgentsResponseManager, MetricStreamBroker
├── core/
│   ├── mongo.py       ManagerBase - подключение MongoClient (connect/disconnect)
│   ├── model.py       CVMetricManager - метрики моделей, parse_split, FINAL_TRAINING_STATUSES
│   ├── agent.py       AgentsResponseManager - метрики агентов, сводка по дискуссии
│   ├── stream.py      MetricStreamBroker (in-memory pub/sub), format_sse
│   ├── stats.py       compute_model_summary, compare_models, is_higher_better (чистые функции)
│   └── health.py      check_health_all (только MongoDB)
└── logs/              логгер (app_json.log)
```
