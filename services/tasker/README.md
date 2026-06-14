# tasker

Сервис управления задачами обучения платформы KiSinWi.
**FastAPI + PostgreSQL** (psycopg2), порт **6110**, БД `tasker_postgres` (порт **6115**).
Зависимости - через **uv** (`uv sync`, `uv run ...`).

> Это документация для разработчика: назначение, запуск, конфиг, HTTP API, модель данных,
> жизненный цикл задачи и взаимодействие с сервисами. Карта модулей `app/` - в конце файла.

## Назначение и место в пайплайне

tasker хранит очередь задач обучения и выдаёт их на исполнение:

```
datasets → agents → [ tasker ] → trainer → metrics + ml_models → agent_history
```

- **agents / frontend** создают задачи (`POST /tasks`) и читают их статус/прогресс;
- **trainer** забирает задачи из очереди (`GET /tasks/next`) и репортит прогресс/итог
  (`POST /tasks/{id}/status`);
- при создании задачи tasker проверяет существование модели в **ml_models**.

В отличие от trainer, у tasker **нет воркера**: это чистый HTTP-сервис над PostgreSQL.

## Архитектура

`main.py` поднимает один uvicorn-сервер. Вся бизнес-логика - в `TrainingTaskManager`
([app/core/train_models_tasks.py](app/core/train_models_tasks.py)), единственном глобальном
экземпляре-синглтоне ([app/api/deps.py](app/api/deps.py)), который делится между всеми запросами.

Слой БД - [app/core/postgresql.py](app/core/postgresql.py): `PostgresManager.session()` выдаёт
сессию-на-запрос (новое соединение → транзакция → `commit`/`rollback` → `close`), строки
читаются как dict через `RealDictCursor`. **Пула соединений нет** (см. [IMPROVEMENTS.md](IMPROVEMENTS.md)).

```
                      ┌──────────────── tasker (main.py, uvicorn) ────────────────┐
                      │                                                            │
  agents/frontend ──▶ │  POST /tasks, GET /tasks, POST /tasks/{id}/cancel, ...     │
                      │        │                                                   │
  trainer ───────────▶│  GET /tasks/next  ─┐                                       │
  trainer ───────────▶│  POST /tasks/{id}/status                                   │
                      │        ▼                                                   │
                      │  TrainingTaskManager ──▶ PostgreSQL (tasker_postgres)      │
                      │        │                                                   │
  ml_models ◀─────────│  GET /versions/{model_id}  (проверка модели при создании)  │
                      └────────────────────────────────────────────────────────────┘
```

## Запуск

```bash
uv sync
python main.py          # uvicorn на 0.0.0.0:6110, reload=True
# либо: uv run uvicorn main:app --host 0.0.0.0 --port 6110
```

В `lifespan` вызывается `check_health_all()` ([app/core/health.py](app/core/health.py)), но он
**не блокирует старт**: сервис поднимется даже при недоступной БД ([main.py:14-16](main.py#L14)),
а запросы будут падать с 503. Порт в `main.py` захардкожен (`6110`); `TASKER_PORT` читается
только в Docker CMD (см. [IMPROVEMENTS.md](IMPROVEMENTS.md)).

### Docker

```dockerfile
FROM python:3.14-slim
# libpq-dev / gcc / python3-dev для сборки psycopg2
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${TASKER_PORT:-6110}"]
```

## Переменные окружения

Читаются в [app/config.py](app/config.py) (`PostgreSQLConfig`, `MLModelsServiceConfig`).

| Переменная               | Назначение                          | По умолчанию       |
|--------------------------|-------------------------------------|--------------------|
| `POSTGRES_HOST`          | хост PostgreSQL                     | `localhost`        |
| `POSTGRES_PORT`          | порт PostgreSQL                     | `6115`             |
| `POSTGRES_APP_USERNAME`  | пользователь БД                     | `tasker_service`   |
| `POSTGRES_APP_PASSWORD`  | пароль БД (**обязательна**)         | нет (RuntimeError при сборке URL) |
| `POSTGRES_DB`            | имя базы                            | `task_service_db`  |
| `ML_MODELS_HOST`         | хост сервиса ml_models              | `localhost`        |
| `ML_MODELS_PORT`         | порт сервиса ml_models              | `6300`             |
| `TASKER_PORT`            | порт сервиса (только в Docker CMD)  | `6110`             |

Без `POSTGRES_APP_PASSWORD` обращение к `postgresql_config.URL` бросает `RuntimeError`
(локально это значение `TASKER_POSTGRES_PASSWORD` из корневого `.env`).

## HTTP API

Роутеры: [app/api/routers/task.py](app/api/routers/task.py) (префикс `/tasks`),
[app/api/routers/info.py](app/api/routers/info.py) (префикс `/info`).

| Метод  | Путь                          | Назначение |
|--------|-------------------------------|------------|
| POST   | `/tasks`                      | создать задачу (`task_name`, `model_id`, опц. `discussion_id`); проверяет модель в ml_models → `{task_id}` |
| GET    | `/tasks`                      | список задач; фильтры `?status=&model_id=&discussion_id=` |
| GET    | `/tasks/next`                 | **атомарно** забрать первую `waiting`-задачу и перевести в `running`; 204 если очередь пуста |
| GET    | `/tasks/count`                | количество задач |
| GET    | `/tasks/{id}`                 | задача по id; 404 если нет |
| DELETE | `/tasks/{id}`                 | удалить задачу; 404 если нет |
| POST   | `/tasks/{id}/status`          | обновить статус/прогресс (`status`, `status_info`, опц. `percentages`, `error`) |
| POST   | `/tasks/{id}/cancel`          | отменить (только из `waiting`/`running`) |
| POST   | `/tasks/{id}/agents-response` | добавить id ответа агента (`?agent_response_id=`, JSONB-append) |
| GET    | `/info/health`               | статус сервиса + зависимостей (БД, ml_models) |
| GET    | `/info/statuses`             | список возможных статусов задачи |

Ключевые для пайплайна:

- **`GET /tasks/next`** - точка входа trainer. Один SQL-запрос атомарно выбирает старейшую
  `waiting`-задачу и переводит её в `running` (`FOR UPDATE SKIP LOCKED`), поэтому два воркера
  не получат одну задачу.
- **`POST /tasks/{id}/status`** - сюда trainer шлёт прогресс и финал. Задача в **финальном**
  статусе (`completed`/`failed`/`cancelled`) **не обновляется** (условный UPDATE), чтобы
  progress-апдейты не перетёрли уже выставленный `cancelled`.

UUID входных id проверяются `valid_uuid` ([app/core/utils.py](app/core/utils.py)). Ошибки БД
ловятся глобальными обработчиками ([app/api/exceptions.py](app/api/exceptions.py)):
`OperationalError`/`InterfaceError` → 503, прочее → 500.

## Модель данных и статусы

DDL: [db/postgres/tasker/init/01-create-tables.sql](../../db/postgres/tasker/init/01-create-tables.sql).
Pydantic-схемы: [app/api/schemas/task.py](app/api/schemas/task.py).

**Таблица `task_statuses`** - справочник из 5 статусов:

| id | status      | смысл |
|----|-------------|-------|
| 1  | `waiting`   | ожидает выполнения (стартовый) |
| 2  | `running`   | выполняется |
| 3  | `completed` | завершено |
| 4  | `failed`    | завершено с ошибкой |
| 5  | `cancelled` | отменена |

**Таблица `train_models_tasks`** (поля задачи):

```jsonc
{
  "id": "uuid",                 // gen_random_uuid()
  "name": "string",             // VARCHAR(50)
  "model_id": "uuid",           // обучаемая модель (ml_models)
  "discussion_id": "uuid|null", // диалог, к которому относится задача
  "agent_respons_ids": [],      // JSONB, id ответов агентов
  "status_id": 1,               // FK → task_statuses
  "percentages": 0,             // прогресс 0..100
  "status_info": "string|null",
  "error_message": "string|null",
  "created_at": "ts",
  "started_at": "ts|null",      // проставляет триггер при переходе в running
  "updated_at": "ts",           // триггер при каждом UPDATE
  "completed_at": "ts|null"     // триггер при переходе в финальный статус
}
```

Индексы: `status_id`, `created_at DESC`. FK `status_id → task_statuses(id)`.
Триггер `handle_task_timestamps` (BEFORE UPDATE): всегда обновляет `updated_at`; ставит
`started_at` при переходе в `running` (id=2); ставит `completed_at` при переходе в финальный
статус (id ∈ 3,4,5).

**Переходы статусов:**

```
            POST /tasks
                │
                ▼
            waiting ──── GET /tasks/next ────▶ running ──┬─ POST /tasks/{id}/status ─▶ completed
                │                                 │      └─ POST /tasks/{id}/status ─▶ failed
                └──── POST /tasks/{id}/cancel ────┴──────── POST /tasks/{id}/cancel ─▶ cancelled
```

Финальные статусы (`completed`/`failed`/`cancelled`) дальше не меняются.

## Жизненный цикл задачи

1. **Создание** (`POST /tasks`): валидация UUID, проверка модели в ml_models
   (`models_is_exists`), `INSERT ... RETURNING id` со статусом `waiting`.
2. **Выдача** (`GET /tasks/next`): атомарный claim - `UPDATE ... WHERE id = (SELECT ... WHERE
   status='waiting' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING id`, затем
   догрузка полей задачи в той же транзакции
   ([train_models_tasks.py:196-233](app/core/train_models_tasks.py#L196)). Триггер ставит `started_at`.
3. **Прогресс / итог** (`POST /tasks/{id}/status`): trainer шлёт `running`+`percentages`, в конце
   `completed`/`failed`. Условный UPDATE не трогает задачу в финальном статусе
   ([train_models_tasks.py:103-136](app/core/train_models_tasks.py#L103)).
4. **Отмена** (`POST /tasks/{id}/cancel`): разрешена только из `waiting`/`running`; ставит
   `cancelled`. trainer ловит это на границе эпох и останавливает обучение.

## Взаимодействие с сервисами

Единственный исходящий вызов - в **ml_models** при создании задачи
([app/service/ml_models/__init__.py](app/service/ml_models/__init__.py)):

| Сервис    | Метод / путь                  | Когда |
|-----------|-------------------------------|-------|
| ml_models | `GET /versions/{model_id}`    | проверка существования модели (`POST /tasks`), timeout 5 c → 503 при недоступности |
| ml_models | `GET /info/health`            | агрегируется в `/info/health` tasker |

Входящие вызовы: от **trainer** (`/tasks/next`, `/tasks/{id}/status`) и **agents/frontend**
(создание, чтение, отмена).

## Нюансы

- **Атомарная выдача задачи.** `GET /tasks/next` одним SQL-запросом выбирает старейшую
  `waiting`-задачу и переводит её в `running` под `FOR UPDATE SKIP LOCKED` - два воркера не
  получат одну задачу.
- **Финальный статус не перетирается.** `POST /tasks/{id}/status` использует условный UPDATE: задача
  в `completed`/`failed`/`cancelled` не обновляется, чтобы progress-апдейты не затёрли `cancelled`.
- **Нет пула соединений.** `PostgresManager.session()` открывает соединение на каждый запрос
  (см. [IMPROVEMENTS.md](IMPROVEMENTS.md)).
- **Старт не блокируется БД.** `check_health_all` в `lifespan` только логирует - сервис поднимется
  при недоступной БД, запросы будут падать с 503.
- **Порт захардкожен** в `main.py` (`6110`); `TASKER_PORT` читается только в Docker CMD.
- **`POSTGRES_APP_PASSWORD` обязателен** - без него `postgresql_config.URL` бросает `RuntimeError`.

## Отладка

- **Логи**: `app/logs/app_json.log`, уровень `DEBUG` ([app/logs/config.py](app/logs/config.py)).
- **Нет `POSTGRES_APP_PASSWORD`** → `RuntimeError` при сборке URL (первое обращение к БД).
- **БД недоступна** → сервис всё равно стартует (health в lifespan не блокирует), запросы
  падают с 503 через exception handlers.
- **ml_models недоступен** → `POST /tasks` отдаёт 503, `/info/health` помечает сервис `degraded`.
- **Финальный статус** → повторный `POST /tasks/{id}/status` вернёт 400 ("задача в финальном
  статусе не может быть обновлена").

## Структура `app/`

```
app/
├── config.py          PostgreSQLConfig, MLModelsServiceConfig (env, URL-ы)
├── api/
│   ├── routers/       task.py (CRUD + next/status/cancel/agents-response), info.py (health, statuses)
│   ├── schemas/       task.py (TaskResponse/Create/Update, ...), info.py (HealthResponse)
│   ├── deps.py        get_training_task_manager (синглтон TrainingTaskManager)
│   └── exceptions.py  обработчики 503 (OperationalError/InterfaceError) и 500
├── core/
│   ├── train_models_tasks.py  TrainingTaskManager - вся бизнес-логика задач
│   ├── postgresql.py          PostgresManager / PostgresSession (сессия-на-запрос)
│   ├── health.py              check_health_all (БД + ml_models)
│   └── utils.py               valid_uuid
├── service/
│   └── ml_models/     models_is_exists, check_health_ml_models
└── logs/              логгер (app_json.log)
```
