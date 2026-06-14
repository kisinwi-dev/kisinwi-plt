# agent_history

Сервис истории рассуждений и действий агентов платформы KiSinWi.
**FastAPI + файловое хранилище** (без БД), порт **6410**, данные в `discussion/`.
Зависимости - через **uv** (`uv sync`, `uv run ...`).

> Это документация для разработчика: назначение, запуск, конфиг, HTTP API, модель данных,
> SSE-поток, раскладка на диске и подводные камни. Карта модулей `app/` - в конце файла.

## Назначение и место в пайплайне

agent_history - финальное звено пайплайна: он хранит «ленту дискуссии» (рассуждения агентов,
их вызовы инструментов и системные сообщения) и отдаёт её фронту, в том числе живым SSE-потоком.

```
datasets → agents → tasker → trainer → metrics + ml_models → [ agent_history ]
```

- **agents** (точнее `development_models`, оттуда же эмитятся системные сообщения) создаёт
  дискуссию (`POST /discussions`), пишет ответы агентов (`POST .../responses`), вызовы
  инструментов (`POST .../tool`), системные сообщения (`POST .../system_messages`) и закрывает
  дискуссию сменой статуса (`PATCH .../meta`);
- **frontend** читает ленту: либо разовыми REST-запросами, либо подпиской на
  `GET /discussions/{id}/stream` (Server-Sent Events).

Источник правды - **файлы на диске**, никакой базы данных у сервиса нет (вопреки тому, что
говорит `/info/health`, см. ниже).

## Архитектура

`main.py` поднимает один uvicorn-сервер (`title="Agents Monitoring"`, CORS открыт на `*`).
Бизнес-логика разнесена по storage-слоям в [app/core/storage/](app/core/storage/), каждый из
которых наследует `BaseStorage` (`base_path="discussion"` относительно CWD). Слои - глобальные
синглтоны в [app/api/deps.py](app/api/deps.py), общие на все запросы:

```
                   ┌──────────── agent_history (main.py, uvicorn) ────────────┐
                   │                                                          │
  agents ────────▶ │  POST /discussions, /responses, /tool, /system_messages  │
                   │  PATCH /discussions/{id}/meta                            │
                   │        │                                                 │
                   │        ▼                                                 │
                   │  *Storage (file I/O)  ──▶  discussion/{id}/... (диск)     │
                   │        │                                                 │
                   │        ▼ publish()                                       │
  frontend ──SSE──▶│  DiscussionStreamBroker (in-memory pub/sub)  ──▶ /stream  │
                   └──────────────────────────────────────────────────────────┘
```

`DiscussionStreamBroker` ([app/core/stream.py](app/core/stream.py)) не передаёт данные, а лишь
будит подписчиков: получив уведомление, поток сам перечитывает полный снимок дискуссии с диска.

## Запуск

```bash
uv sync
python main.py          # uvicorn на 0.0.0.0:6410, reload=True
# либо: uv run uvicorn main:app --host 0.0.0.0 --port 6410
```

Порт в [main.py](main.py#L29) **захардкожен** (`6410`). Health-проверки в `lifespan` нет.

### Docker

```dockerfile
FROM python:3.14-slim
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${AGENT_MONITORING_PORT:-6410}"]
```

Данные монтируются томом `./db/agent_history/discussion:/app/discussion` (см.
[docker-compose.yml](../../docker-compose.yml)) - именно сюда складываются каталоги дискуссий.

## Переменные окружения

Отдельного config-модуля у сервиса **нет**. Единственная переменная читается только в Docker CMD.

| Переменная               | Назначение                          | По умолчанию |
|--------------------------|-------------------------------------|--------------|
| `AGENT_MONITORING_PORT`  | порт сервиса (**только** в Docker CMD; в `main.py` игнорируется) | `6410` |

## HTTP API

Роутеры собираются в [app/api/routers/__init__.py](app/api/routers/__init__.py).

| Метод  | Путь                                                  | Назначение |
|--------|-------------------------------------------------------|------------|
| POST   | `/discussions`                                        | создать дискуссию (`discussion_id` опц.: если не передан - `uuid4`) → `DiscussionMeta` |
| GET    | `/discussions`                                        | список дискуссий с агрегатами; фильтры `?status=&pipeline=&skip=&limit=` |
| GET    | `/discussions/{id}/meta`                              | метаданные дискуссии; 404 если нет |
| PATCH  | `/discussions/{id}/meta`                              | обновить мету (title/status/tags/pipeline/agent_roles); публикует SSE |
| GET    | `/discussions/{id}/stream`                            | **SSE-поток** ленты дискуссии (см. ниже) |
| DELETE | `/discussions/{id}`                                   | удалить дискуссию (рекурсивно весь каталог); 404 если нет |
| POST   | `/discussions/{id}/responses`                         | сохранить ответ агента (`AgentResponse`); публикует SSE |
| GET    | `/discussions/{id}/responses`                         | ответы агентов, отсортированы по времени; 404 если дискуссии нет |
| POST   | `/discussions/{id}/system_messages`                   | сохранить системное сообщение; публикует SSE |
| GET    | `/discussions/{id}/system_messages`                   | системные сообщения, по времени; 404 если дискуссии нет |
| POST   | `/discussions/{id}/tool`                              | сохранить вызов инструмента (`Tool`); **SSE НЕ публикует** |
| GET    | `/discussions/{id}/responses/{response_id}/tools`     | вызовы инструментов конкретного запуска агента |
| GET    | `/info/health`                                        | статус сервиса (см. подводные камни) |
| GET    | `/info/type_system_mes`                               | список значений `SystemMessageType` |
| GET    | `/info/status_tool`                                   | список значений `ToolStatus` |
| GET    | `/info/status_agent`                                  | список значений `AgentStatus` |

### SSE-поток `/discussions/{id}/stream`

[discussions.py](app/api/routers/discussions.py) отдаёт `text/event-stream`:

- при подключении и после каждого `publish` шлёт событие **`discussion`** с полным снимком
  `DiscussionSnapshot` (мета + responses + system_messages, отсортированные по времени);
- при финальном статусе (`completed`/`failed`) после снимка один раз идёт событие **`end`**
  с телом `{discussion_id, status}`; подписчику уже завершённой дискуссии снимок и `end`
  приходят сразу при подключении;
- после `end` поток сервером **не закрывается** - иначе браузерный `EventSource` уходит в
  бесконечный реконнект; соединение закрывает клиент;
- каждые **15 секунд** отправляется keepalive-комментарий (`: ping`), чтобы прокси не рвали
  соединение; заголовок `X-Accel-Buffering: no` отключает буферизацию nginx;
- **подключаться можно к ещё не созданной дискуссии** - снимок придёт с `meta=null`;
- накопившиеся уведомления схлопываются (снимок всё равно полный), а переполнение очереди
  подписчика игнорируется - актуальность даёт повторное чтение с диска.

**Важно:** вызовы инструментов (`tool`) в снимок **не входят** и в ленту по SSE не попадают -
их забирают отдельным `GET .../responses/{response_id}/tools`.

## Модель данных и схемы

Pydantic-схемы - в [app/api/schemas/](app/api/schemas/). Независимые наборы enum-статусов:

| Enum                | Значения | Где |
|---------------------|----------|-----|
| `DiscussionStatus`  | `active`, `completed`, `failed` | статус дискуссии |
| `AgentStatus`       | `IN PROGRESS`, `SUCCEED`, `ERROR` | статус ответа агента |
| `ToolStatus`        | `IN PROGRESS`, `SUCCEED`, `ERROR` | статус вызова инструмента |
| `SystemMessageType` | `INFO`, `WARNING`, `ERROR` | тип системного сообщения |

> Внимание: у `AgentStatus`/`ToolStatus` имя члена `SUCCEEDED`, а его value - строка `"SUCCEED"`
> (по API ходит именно `SUCCEED`).

Ключевые модели:

- **`DiscussionMeta`** ([discussion_meta.py](app/api/schemas/discussion_meta.py)): `discussion_id`,
  `title`, `status`, `tags`, `pipeline`, `agent_roles`, `created_at`, `finished_at`.
  `finished_at` проставляется автоматически при переходе в `completed`/`failed` (один раз).
- **`DiscussionMetaRead`** (для списка): `DiscussionMeta` + вычисляемые `responses_count`,
  `tool_calls_count`, `agents` (роли и использованные ими модели LLM).
- **`AgentResponse`** ([agent.py](app/api/schemas/agent.py)): `response_id`, `agent_role`,
  `text` (min_length=1), `model`, `duration_ms`, `task_name`, `iteration`, `timestamp`.
- **`Tool`** ([tool.py](app/api/schemas/tool.py)): `id`, `agent_role`, `name`, `status`, `message`,
  `input_args`, `output`, `error_traceback`, `response_id` (привязка к запуску агента).
- **`SystemMessage`** ([system_message.py](app/api/schemas/system_message.py)): `type_`, `message`, `timestamp`.
- **`DiscussionSnapshot`** ([stream.py](app/api/schemas/stream.py)): тело SSE-события `discussion`.

## Хранилище на диске

Никакого индекса/БД - всё лежит файлами под `discussion/{discussion_id}/`:

```
discussion/
└── {discussion_id}/
    ├── meta.json                     # DiscussionMeta
    ├── responses/
    │   └── {response_id}.json         # AgentResponse (имя файла = response_id)
    ├── system/
    │   └── {uuid4}.json               # SystemMessage (имя файла случайное)
    └── tools/
        └── {response_id}/             # вызовы инструментов одного запуска агента
            └── {tool_id}.json         # Tool
```

Особенности записи:

- **responses и tools идемпотентны:** файл называется по своему id (`response_id` / `tool_id`),
  повторный POST = перезапись (upsert). `ToolStorage.update` - это просто алиас `save`.
- **system_messages НЕ идемпотентны:** имя файла - случайный `uuid4`, поэтому повторный POST того
  же сообщения создаёт **дубликат**.
- **tool без `response_id`** уходит в каталог `tools/_unlinked/`
  ([tool.py](app/core/storage/tool.py)).
- `GET /discussions` (список) на каждый вызов обходит все каталоги дискуссий и читает каждый файл
  ответа, считая агрегаты - это **O(n) скан ФС без кэша**. В `agents` сперва идут роли,
  объявленные в `meta.agent_roles`, затем реально отвечавшие, но не объявленные
  ([discussion.py](app/core/storage/discussion.py)).

## Взаимодействие с сервисами

agent_history - финальное звено: исходящих вызовов к другим сервисам у него нет, только
локальная файловая система. Все вызовы входящие:

| Источник | Метод / путь | Когда |
|----------|--------------|-------|
| agents (`development_models`) | `POST /discussions` | создание дискуссии в начале пайплайна |
| agents (`development_models`) | `POST .../responses`, `.../tool`, `.../system_messages` | запись ответов агентов, вызовов инструментов, системных сообщений |
| agents (`development_models`) | `PATCH /discussions/{id}/meta` | обновление меты, закрытие дискуссии сменой статуса |
| frontend | `GET /discussions*` (REST) + `GET /discussions/{id}/stream` (SSE) | чтение ленты разовыми запросами и живым потоком |

`GET /info/health` ничего не проверяет (заглушка, БД у сервиса нет, см. «Нюансы»).

## Нюансы

- **SSE работает только в одном процессе uvicorn.** `DiscussionStreamBroker` - in-memory pub/sub;
  при нескольких воркерах издатель и подписчики окажутся в разных процессах, и живая лента
  перестанет обновляться (REST-чтение при этом продолжит работать).
- **Вызовы инструментов не попадают в SSE-ленту** (нет `publish` в `POST .../tool` и нет их в
  снимке) - забирать отдельным GET по `response_id`.
- **responses и tools идемпотентны** (имя файла = свой id, повторный POST = перезапись), а
  **system_messages - нет** (имя файла случайный `uuid4`), поэтому повторная отправка одного и
  того же сообщения плодит **дубли**.
- **`GET /discussions`** на каждый вызов обходит все каталоги дискуссий и читает каждый файл
  ответа - это O(n) скан ФС без кэша.
- **`DiscussionStorage.get_history`** (сводная лента system+responses+tools по timestamp) написан,
  но **не подключён ни к одному роутеру** - мёртвый код.

## Отладка

- **Логи**: `app/logs/app_json.log` (см. [app/logs/](app/logs/)).
- **`/info/health` всегда `healthy`** → проверка захардкожена в [health.py](app/core/health.py)
  (`db: healthy`), реальной проверки нет, ветка `degraded` недостижима; БД у сервиса нет.
- **Живая лента не обновляется** → запущено несколько воркеров uvicorn (in-memory брокер не
  пересекает процессы); запускать в один процесс.
- **Дубликаты системных сообщений** → повторный `POST .../system_messages` (неидемпотентен, см. «Нюансы»).
- **Вызова инструмента нет в живой ленте** → `tool` не публикуется в SSE; забирать
  `GET .../responses/{response_id}/tools`.

## Структура `app/`

```
app/
├── api/
│   ├── routers/
│   │   ├── discussions.py     CRUD дискуссий + PATCH meta + SSE /stream
│   │   ├── responses.py       сохранение/чтение ответов агентов
│   │   ├── agent_system.py    системные сообщения
│   │   ├── agent_tools.py     вызовы инструментов
│   │   └── info.py            health + справочники enum-статусов
│   ├── schemas/               discussion_meta, agent, tool, system_message, stream, info
│   └── deps.py                синглтоны storage-слоёв и broker
├── core/
│   ├── storage/
│   │   ├── base.py            BaseStorage (base_path="discussion")
│   │   ├── discussion.py      meta, список с агрегатами, get_history (не используется)
│   │   ├── response.py        ответы агентов
│   │   ├── system.py          системные сообщения
│   │   └── tool.py            вызовы инструментов (+ _unlinked)
│   ├── stream.py              DiscussionStreamBroker, format_sse
│   └── health.py              check_health_all (заглушка)
└── logs/                      логгер (app_json.log)
```
