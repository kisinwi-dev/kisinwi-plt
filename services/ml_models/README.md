# ml_models

Сервис-реестр ML-моделей и файлов их весов платформы KiSinWi.
**FastAPI + PostgreSQL** (psycopg2), порт **6300**, БД `ml_models_postgres`
(порт **6305**, база `ml_models_service_db`).
Зависимости - через **uv** (`uv sync`, `uv run ...`).

> Это документация для разработчика: назначение, запуск, конфиг, HTTP API, модель
> данных, хранение весов, взаимодействие с сервисами и неочевидные детали реализации.
> Карта модулей `app/` - в конце файла.

## Назначение и место в пайплайне

ml_models хранит обученные модели как реестр и отдаёт их веса:

```
datasets → agents → tasker → trainer → metrics + [ ml_models ] → agent_history
                                ↑ статус + ONNX-веса
```

Три сущности (три таблицы):

- **модель** - родитель: уникальное имя + описание (`models`);
- **версия** - конкретный обученный артефакт модели: статус, датасет, классы,
  параметры обучения (`model_versions`);
- **файл весов** - ONNX-файл, привязанный к версии (`ml_model_files`).

Кто и как пользуется сервисом:

- **trainer** забирает `train_params` (`GET /versions/{id}`), двигает статус
  (`PATCH /versions/{id}`) и заливает ONNX-веса (`POST /versions/{id}/files`);
- **tasker** проверяет существование версии при создании задачи (`GET /versions/{id}`);
- **frontend / agents** читают модели, версии, статистику и скачивают веса.

## Архитектура

`main.py` поднимает один uvicorn-сервер. Бизнес-логика - в трёх менеджерах:
`ModelsManager` ([app/core/models.py](app/core/models.py)),
`VersionsManager` ([app/core/versions.py](app/core/versions.py)),
`FilesManager` ([app/core/files.py](app/core/files.py)). Все три - глобальные
синглтоны, создаются при импорте [app/api/deps.py](app/api/deps.py) и делятся между
запросами.

Слой БД - [app/core/postresql.py](app/core/postresql.py) (имя файла именно
`postresql.py`, без второй `g`). В отличие от tasker, здесь **есть пул соединений**:
`ThreadedConnectionPool` создаётся один раз на класс (`maxconn=10`), а само
соединение/курсор хранятся в `threading.local`, чтобы sync-роуты FastAPI из threadpool
не делили один курсор. Интерфейс - `with self.db as db: db.fetch_all(...)`.

```
                      ┌──────────────── ml_models (main.py, uvicorn) ─────────────┐
                      │                                                           │
  frontend/agents ──▶ │  GET /models, GET /models/{id}, GET /models/statistics    │
                      │                                                           │
  trainer  ──────────▶│  GET /versions/{id}, PATCH /versions/{id}                 │
  trainer  ──────────▶│  POST /versions/{id}/files (ONNX)                         │
                      │        │                                                  │
  tasker  ───────────▶│  GET /versions/{id} (проверка модели)                     │
                      │        ▼                                                  │
                      │  Models/Versions/FilesManager ──▶ PostgreSQL (pool)       │
                      │                              └──▶ диск: model_files/       │
                      └─────────────────────────────────────────────────────────┘
```

## Запуск

```bash
uv sync
python main.py          # uvicorn на 0.0.0.0:6300, reload=True
# либо: uv run uvicorn main:app --host 0.0.0.0 --port 6300
```

В `__main__` вызывается `check_health_all()` ([app/core/health.py](app/core/health.py)),
но он **не блокирует старт**: сервис поднимется даже при недоступной БД
([main.py:32-42](main.py#L32)), а запросы будут падать с 503. Порт в `main.py`
захардкожен (`6300`); `ML_MODELS_SERVICE_PORT` читается только в Docker CMD.

### Docker

```dockerfile
FROM python:3.14-slim
# libpq-dev / gcc / python3-dev для сборки psycopg2
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${ML_MODELS_SERVICE_PORT:-6300}"]
```

Файлы весов лежат на томе `./db/model_files:/app/model_files` (директория `model_files`
в корне сервиса).

## Переменные окружения

Читаются в [app/config.py](app/config.py) (`PostgreSQLConfig`).

| Переменная                | Назначение                          | По умолчанию          |
|---------------------------|-------------------------------------|-----------------------|
| `POSTGRES_HOST`           | хост PostgreSQL                     | `localhost`           |
| `POSTGRES_PORT`           | порт PostgreSQL                     | `6305`                |
| `POSTGRES_APP_USERNAME`   | пользователь БД                     | `ml_models_service`   |
| `POSTGRES_APP_PASSWORD`   | пароль БД                           | `060720` (небезопасный дефолт) |
| `POSTGRES_DB`             | имя базы                            | `ml_models_service_db`|
| `ML_MODELS_SERVICE_PORT`  | порт сервиса (только в Docker CMD)  | `6300`                |

Другие сервисы обращаются к ml_models по своим переменным (`ML_MODELS_PORT` у tasker,
`ML_MODELS_DOMAIN` у trainer) - в самом ml_models этих переменных нет.

## HTTP API

Роутеры: [app/api/routers/ml_models.py](app/api/routers/ml_models.py) (префикс `/models`),
[app/api/routers/versions.py](app/api/routers/versions.py),
[app/api/routers/files.py](app/api/routers/files.py),
[app/api/routers/info.py](app/api/routers/info.py) (префикс `/info`).

### Модели (`/models`)

| Метод  | Путь                  | Назначение |
|--------|-----------------------|------------|
| GET    | `/models`             | модели с вложенными версиями; фильтры `?name=&status=&dataset_id=` + пагинация `limit/offset` |
| POST   | `/models`             | создать модель (`name`, опц. `description`) → `{model_id}`; 409 если имя занято |
| GET    | `/models/statistics`  | счётчики: всего моделей, всего версий, версий по статусам |
| GET    | `/models/by-name/{name}` | модель по точному имени; 404 если нет |
| GET    | `/models/{model_id}`  | модель с версиями (версии по убыванию); 404 если нет |
| PATCH  | `/models/{model_id}`  | обновить `name`/`description`; 409 при конфликте имени |
| DELETE | `/models/{model_id}`  | удалить модель со всеми версиями и файлами → 204 |

### Версии (`/versions`)

| Метод  | Путь                          | Назначение |
|--------|-------------------------------|------------|
| POST   | `/models/{model_id}/versions` | создать версию; **номер назначает сервер** (MAX+1) → `{version_id, version}` |
| GET    | `/versions`                   | плоский список версий (свежие сверху); фильтры `?name=&status=&dataset_id=&model_id=` + пагинация, в ответе `total` |
| GET    | `/versions/{version_id}`      | одна версия (с именем/описанием родителя); 404 если нет |
| PATCH  | `/versions/{version_id}`      | частичное обновление (`status`, `metrics_report`, `train_params`, `framework`, ...) |
| DELETE | `/versions/{version_id}`      | удалить версию вместе с файлами → 204 |

### Файлы весов

| Метод  | Путь                          | Назначение |
|--------|-------------------------------|------------|
| GET    | `/versions/{version_id}/files`| список файлов версии; 204 если файлов нет; 404 если нет версии |
| POST   | `/versions/{version_id}/files`| загрузить файл (multipart `UploadFile`); 409 если файл уже есть |
| DELETE | `/versions/{version_id}/files`| удалить файлы по `ids` из тела (или все, если `ids` не передан) |
| GET    | `/files/{file_id}/download`   | скачать файл (`FileResponse`, `application/octet-stream`) |

### Информация (`/info`)

| Метод | Путь                  | Назначение |
|-------|-----------------------|------------|
| GET   | `/info/health`        | статус сервиса + БД (`healthy`/`degraded`) |
| GET   | `/info/models/status` | справочник статусов модели с описаниями |

UUID из пути проверяются зависимостями `validate_model_id` / `validate_version_id` /
`validate_file_id` ([app/api/deps.py](app/api/deps.py)) - невалидный id даёт 404, а не
422. Ошибки БД ловятся глобальными обработчиками
([app/api/exceptions.py](app/api/exceptions.py)): `OperationalError`/`InterfaceError`
→ 503, прочее → 500.

## Модель данных и статусы

DDL: [db/postgres/ml_models/init/01-create-table-ml_models.sql](../../db/postgres/ml_models/init/01-create-table-ml_models.sql).
Pydantic-схемы: [app/api/schemas/](app/api/schemas/).

**Справочник `ml_model_statuses`** - 3 статуса:

| id | status      | смысл |
|----|-------------|-------|
| 1  | `draft`     | не обучена (стартовый) |
| 2  | `training`  | в процессе обучения |
| 3  | `completed` | обучена |

Жизненный цикл версии задаёт **trainer**: `draft → training → completed`, с откатом в
`draft` при ошибке или отмене обучения.

**`models`** - родитель: `id` (UUID), `name` (UNIQUE), `description`, `created_at`.

**`model_versions`** - версия:

```jsonc
{
  "id": "uuid",                 // gen_random_uuid(), глобальный идентификатор версии
  "model_id": "uuid",           // FK → models(id) ON DELETE CASCADE
  "version": 1,                 // INTEGER, порядковый номер внутри модели (>= 1)
  "model_type": "resnet50",     // VARCHAR(100), NOT NULL
  "status_id": 1,               // FK → ml_model_statuses(id), DEFAULT 1 (draft)
  "metrics_report": "No info",  // TEXT, свободный отчёт
  "classes": ["cat", "dog"],    // JSONB, NOT NULL, непустой (CHECK length > 0)
  "dataset_id": "uuid",         // VARCHAR(36), NOT NULL
  "dataset_version_id": "uuid", // VARCHAR(36), NOT NULL
  "framework": "onnx",          // VARCHAR(50)
  "framework_version": "1.x",   // VARCHAR(20)
  "train_params": { },          // JSONB, NOT NULL, полный конфиг обучения
  "created_at": "ts"
}
```

Constraints: `unique_model_version (model_id, version)`, `check_classes_not_empty`,
FK на статус и на родителя. Индексы: `(model_id, version DESC)`,
`(dataset_id, dataset_version_id)`, `(status_id)`.

**`ml_model_files`** - метаданные файла: `id` (UUID), `version_id`
(FK → model_versions, `ON DELETE CASCADE`), `filename`, `file_path` (относительно
`storage_dir`), `file_size`, `created_at`. Индекс по `version_id`.

## Хранение весов

Сами ONNX-файлы лежат **на диске**, в БД - только метаданные.

```
model_files/                 # storage_dir (том db/model_files/)
└── {version_id}/
    └── {filename}           # например onnx_model.onnx
```

При загрузке имя файла санитизируется до basename
([app/core/files.py:27-29](app/core/files.py#L27)); `file_path` пишется в БД
относительным (устойчив к переезду тома), `file_size` берётся со стороны диска.

## Взаимодействие с сервисами

ml_models - чистый реестр: **исходящих** вызовов к другим сервисам у него нет, только
БД и диск. Входящие:

| Откуда            | Метод / путь                      | Когда |
|-------------------|-----------------------------------|-------|
| trainer           | `GET /versions/{id}`              | забрать `train_params` |
| trainer           | `PATCH /versions/{id}`           | статус модели (`training`/`completed`/`draft`) |
| trainer           | `POST /versions/{id}/files`      | upload ONNX (multipart) |
| tasker            | `GET /versions/{id}`             | проверка существования модели при `POST /tasks` |
| frontend / agents | `GET /models*`, `/files/{id}/download` | чтение и скачивание весов |

## Нюансы

Неочевидные детали, которые не видны при беглом чтении:

- **Номер версии генерирует сервер прямо в INSERT**:
  `(SELECT COALESCE(MAX(version), 0) + 1 ...)`. При гонке двух
  `POST /models/{id}/versions` ловится `unique_model_version`-конфликт и делается
  **ровно один** retry ([app/core/versions.py:110-134](app/core/versions.py#L110)) -
  при высокой параллельности окно остаётся, второй конфликт уйдёт в 500.
- **У версии два идентификатора**: `id` (UUID, глобальный) и `version` (INTEGER,
  порядковый внутри модели). Для адресации в API используется `id`, `version` - только
  для отображения ("версия 3").
- **Денормализация name/description**: ответ `ModelVersion` несёт имя и описание
  родителя ([app/core/versions.py](app/core/versions.py)), но при `PATCH /models/{id}`
  версии **не переписываются** - версия ведёт себя как снимок.
- **`classes` обязателен и непуст**: `NOT NULL` + `CHECK jsonb_array_length > 0`.
  Создать версию без списка классов нельзя - INSERT упадёт по constraint.
- **Удаление двухфазное и без транзакции**: сначала DELETE в БД (FK CASCADE сносит
  версии и метаданные файлов), затем вручную `drop_version_dir` чистит папки с диска
  ([ml_models.py:202-214](app/api/routers/ml_models.py#L202),
  [versions.py:174-190](app/api/routers/versions.py#L174)). Падение между шагами оставит
  orphaned-файлы на диске.
- **При download путь файла реконструируется** из `storage_dir/version_id/filename`,
  а колонка `file_path` из БД при этом не используется
  ([app/core/files.py:241-263](app/core/files.py#L241)). То же в `drop`.
- **Пул соединений**: `ThreadedConnectionPool` class-level, соединение/курсор в
  thread-local, `maxconn=10` ([app/core/postresql.py](app/core/postresql.py)). При >10
  одновременных запросах - блокировка на `getconn()`; явного `closeall()` при shutdown
  нет.
- **`GET /models`: JOIN vs LEFT JOIN** переключается от наличия фильтров по версиям
  (`status`/`dataset_id`): с фильтром берётся INNER JOIN и модели без подходящих версий
  выпадают; без фильтра - LEFT JOIN ([app/core/models.py](app/core/models.py)).
- **Сортировка**: модели отдаются по `name ASC`, версии - по `version DESC`, не по
  `created_at`. Пагинация по моделям идёт по имени.
- **Старт не блокируется** при недоступной БД (`check_health_all` только логирует),
  **дефолтный пароль** `060720` зашит в [app/config.py](app/config.py), а глобальный
  обработчик перехватывает любой `Exception` → 500.

## Отладка

- **Логи**: `app/logs/app_json.log`, уровень `DEBUG` ([app/logs/config.py](app/logs/config.py)).
- **БД недоступна** → сервис всё равно стартует (health не блокирует), запросы падают
  с 503 через exception handlers; `/info/health` помечает БД `unhealthy`, статус сервиса
  `degraded`.
- **Невалидный UUID в пути** → 404 (через `validate_*_id`), не 422.
- **Повторная загрузка файла с тем же именем** → 409 (`FileExistsError`).
- **Файл есть в БД, но нет на диске** → `GET /files/{id}/download` отдаёт 404
  ("Физический файл не найден").

## Структура `app/`

```
app/
├── config.py          PostgreSQLConfig (env, URL)
├── api/
│   ├── routers/       ml_models.py (CRUD моделей + statistics + by-name),
│   │                  versions.py (CRUD версий), files.py (файлы + download), info.py
│   ├── schemas/       ml_models.py (Model/ModelVersion), files.py, info.py
│   ├── deps.py        синглтоны менеджеров + validate_*_id
│   └── exceptions.py  обработчики 503 (OperationalError/InterfaceError) и 500
├── core/
│   ├── models.py      ModelsManager (родители, get_statistics)
│   ├── versions.py    VersionsManager (версии, get_statuses_info)
│   ├── files.py       FilesManager (диск + метаданные файлов)
│   ├── postresql.py   PostgresManager (пул соединений, thread-local)
│   ├── health.py      check_connection_status / check_health_all
│   └── utils.py       valid_uuid
└── logs/              логгер (app_json.log)
```
