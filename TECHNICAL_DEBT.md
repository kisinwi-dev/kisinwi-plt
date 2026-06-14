# Technical Debt

Last updated: 2026-06-13
Source: multi-agent audit (`/audit-project`), scope = весь проект, режим findings-only.

## Summary

**Total: 41** | Critical: 1 | High: 13 | Medium: 13 | Low: 14

Чисто (проверено, проблем нет): SQL/NoSQL-инъекции, zip-slip, небезопасная десериализация, command injection, XSS на фронте, `any` на фронте. Дублирования бизнес-логики между quick/full пайплайнами нет.

---

## Critical

### [ ] C1. Живые секреты в .env
- **File**: `services/agents/.env:3`, `services/trainer/.env:1`
- Рабочие `OPENROUTER_API_KEY` и `HF_TOKEN` на диске. Игнорятся git, но утекут в любой образ/бэкап.
- **Fix**: ротировать оба ключа немедленно; инжектить через секрет-менеджер. **Effort**: small

---

## High

### [ ] H1. Пароли БД в закоммиченном root .env
- **File**: `.env:30-31,52-53,72-73`
- `.env` отслеживается git (нет в `.gitignore`). Все пароли Postgres/Mongo (`060720`, `qwry123`, переиспользованы) в истории.
- **Fix**: `git rm --cached .env`, добавить в `.gitignore`, ротировать пароли, не публиковать порты БД на хост без нужды. **Effort**: medium

### [ ] H2. Path traversal через id_data при загрузке датасета
- **File**: `services/datasets/app/api/routers/upload.py:33-37`
- Невалидированный Form-параметр `id_data` идёт в имя файла/путь, `../../app/evil` пишет за пределы temp.
- **Fix**: валидировать по `^[A-Za-z0-9_-]+$` или `Path(id_data).name`. **Effort**: small

### [ ] H3. Path traversal через discussion_id в agent_history
- **File**: `services/agent_history/app/core/storage/discussion.py:46,174,182` (+ system.py:16, response.py:15, tool.py)
- `discussion_id` из URL без проверки join-ится в путь для read/write/delete.
- **Fix**: dependency с проверкой UUID (переиспользовать паттерн `valid_uuid` из ml_models). **Effort**: small

### [ ] H4. Бесконечный poll-цикл без таймаута
- **File**: `services/agents/app/services/tasker/client.py:77-86`
- `waiting_completed`: `while True` + `sleep(2)`, нет дедлайна. Зависшая задача вешает поток пайплайна навсегда.
- **Fix**: max-wait/max-iterations, при превышении — failure + пометить discussion failed; интервал в именованную константу. **Effort**: small

### [ ] H5. HTTP-вызовы без timeout (get_json/post_json)
- **File**: `services/agents/app/services/utils/get.py:45-60`
- Нет `timeout`; зависший datasets/ml_models блокирует поток пайплайна (хот-путь).
- **Fix**: добавить `timeout=` (connect/read tuple). **Effort**: small

### [ ] H6. HTTP-вызов без timeout (agent_history client)
- **File**: `services/agents/app/services/agent_history/client.py:34`
- Fire-and-forget history-клиент без `timeout`, зовётся десятки раз за пайплайн.
- **Fix**: добавить `timeout=` (broad except уже на месте). **Effort**: small

### [ ] H7. Silent error propagation в промпт LLM
- **File**: `services/agents/app/services/datasets/utils.py:12-37` → `core/quick_pipeline.py:16-24`, `_pipeline_common.py:53-59`
- `@handle_errors` возвращает `{"ERROR": ...}`, `_build_dataset_info` сериализует это прямо в промпт ML-инженера — агент проектирует обучение по блобу с ошибкой.
- **Fix**: проверять `"ERROR" in result` и падать рано; лучше — типизированное исключение вместо sentinel-dict. **Effort**: medium

### [ ] H8. Блокирующий psycopg2 + requests в async-эндпоинтах tasker
- **File**: `services/tasker/app/api/routers/task.py` (все хендлеры `async def`) + `core/postgresql.py:64` + `service/ml_models/__init__.py:12`
- Синхронный psycopg2 + `requests` на event loop сериализуют весь сервис (включая claim воркера).
- **Fix**: сделать хендлеры `def` или `run_in_threadpool`. **Effort**: medium

### [ ] H9. Синхронный pymongo в async-хендлерах metrics
- **File**: `services/metrics/app/api/routers/model.py:55,77,96,...` + `agent.py:27,54,75,100`
- `async def` + блокирующий pymongo; trainer пушит метрики каждую эпоху — сервис стопорится.
- **Fix**: `def` / `asyncio.to_thread` / motor. **Effort**: medium

### [ ] H10. MongoClient создаётся/закрывается на каждый запрос
- **File**: `services/metrics/app/core/mongo.py:24-51`, `api/deps.py:6-24`
- `with manager:` зовёт connect/disconnect per request, убивая внутренний пул.
- **Fix**: открыть клиент один раз в lifespan, не закрывать per-request. **Effort**: medium

### [ ] H11. tasker открывает новое соединение на каждый запрос (без пула)
- **File**: `services/tasker/app/core/postgresql.py:60-76`
- ml_models использует `ThreadedConnectionPool`, tasker — `psycopg2.connect()` на каждый запрос (хот-путь `/tasks/next`).
- **Fix**: `ThreadedConnectionPool`, переиспользовать пул из ml_models. **Effort**: medium

### [ ] H12. Нет тестов: pipeline orchestration
- **File**: `services/agents/app/core/pipeline/train.py`, `core/quick_pipeline.py`, `api/routers/_pipeline_common.py`
- 0 тестов; ветки cancel-vs-fail, `{"ERROR":...}` sentinel, partial-metrics, lifecycle `models_context`.
- **Fix**: unit-тесты с мок-клиентами: success, create-version error, user-cancel, training-fail. **Effort**: medium

### [ ] H13. Нет тестов: task state machine + trainer worker
- **File**: `services/tasker/app/core/train_models_tasks.py`, `services/trainer/app/core/worker.py`
- Инвариант финальных статусов (`update_status`), атомарный claim (`FOR UPDATE SKIP LOCKED`), orphan-recovery, OOM, rollback — не покрыто.
- **Fix**: интеграционные тесты на блокировку progress после финала, concurrent claim, orphan-recovery, OOM, fail+rollback. **Effort**: medium

---

## Medium

- [ ] M1. CORS `allow_origins=["*"]` + `allow_credentials=True` на всех сервисах (agents/datasets/metrics/agent_history/ml_models/tasker `main.py`). Auth нет — любая страница может управлять платформой. Fix: явный allowlist origin. small
- [ ] M2. Нет retry в poll-цикле — транзиентная сетевая ошибка убивает пайплайн. `tasker/client.py:69-86`. Fix: try/except в теле цикла + счётчик подряд-фейлов. small
- [ ] M3. Общий `requests.Session` (singleton-клиенты) между конкурентными фоновыми пайплайнами — не thread-safe. `utils/client.py:14`. Fix: session per call / lock / документировать контракт. medium
- [ ] M4. metrics `_post` без timeout. `agents/.../metrics/client.py:43-46`. small
- [ ] M5. `models_context.get_models()[-1]` вместо `training_res.version_id` — лишняя связанность + риск IndexError. `core/quick_pipeline.py:130`. small
- [ ] M6. `GET /tasks` без пагинации, грузит всю таблицу. `tasker/.../task.py:64-82`, `train_models_tasks.py:155-184`. Fix: limit/offset. small
- [ ] M7. Нет индекса на `discussion_id` в Mongo (full scan). `metrics/app/core/agent.py:12-17,57-78`. Fix: `create_index('discussion_id')`. small
- [ ] M8. Chatty write-путь метрик: 1+2N round-trips/эпоху + TOCTOU exists-check. `metrics/app/core/model.py:99-156`. Fix: upsert с `$push` / `bulk_write`. medium
- [ ] M9. Загрузка датасета схлопывает все ошибки в HTTP 500 (ValueError/FileNotFoundError = ввод клиента). `datasets/.../upload.py:38-47`. Fix: 400/415 для них. small
- [ ] M10. copy-ID `<span onClick>` недоступен с клавиатуры (4 места). `ModelDetail.tsx:236`, DiscussionCard/DatasetCard/DatasetDetail. Fix: `<button>` или role+tabIndex+onKeyDown. small
- [ ] M11. FileUploader drop-zone mouse-only. `FileUploader.tsx:85-92`. Fix: role/tabIndex/onKeyDown/aria-label. small
- [ ] M12. Нет session-reuse в get.py (новое соединение per call). `agents/.../utils/get.py:45-60`. small
- [ ] M13. Нет тестов на inter-service HTTP-клиенты (status mapping, None/sentinel-контракты). `agents/.../tasker/client.py`, `trainer/.../tasker.py`, `ml_models/history.py`. medium

---

## Low

- [ ] L1. `reload=True` в проде во всех `main.py`. Fix: гнать reload только из dev-compose. small
- [ ] L2. TOCTOU + межсервисный HTTP в роутере `create_task` (нарушает «логика в core»). `tasker/.../task.py:41-53`. small
- [ ] L3. `result is None` трактуется как `failed`, хотя отмена пользователем без метрик легитимна. `_pipeline_common.py:70-73`. Fix: status-enum. medium
- [ ] L4. datasets list читает/валидирует метаданные всех датасетов до пагинации. `datasets/app/core/services/dataset.py:290-308`. medium
- [ ] L5. Mongo `find` с `$in` без projection (грузит все splits). `metrics/app/core/model.py:197-207`. medium
- [ ] L6. `key={index}` в редактируемых/reorderable списках. `SourcesEditor.tsx:94`, ModelCompare:291, VersionStatsModal:421, DatasetDetail:391. small
- [ ] L7. `NotificationContext` value без `useMemo` — лишние ре-рендеры. `NotificationContext.tsx:53`. small
- [ ] L8. Бандл 944 kB одним чанком, без code-splitting. `App.tsx` / `vite.config.ts`. Fix: route-level `React.lazy`. medium
- [ ] L9. `handleResponse` без рантайм-валидации (`JSON.parse as T`, `true as T` для пустого тела). `services/http.ts:29-30`. medium
- [ ] L10. Хардкод `http://` во фронте (нет TLS). `services/http.ts:8-9`. small
- [ ] L11. Дублированный `valid_uuid` (tasker + ml_models) + устаревший docstring (`raise_on_error` → `on_error`). `ml_models/app/core/utils.py:1-23`. small
- [ ] L12. Копипаста `ml_engeneer_quick/tools.py` (байт-в-байт с `ml_engeneer/tools.py`). Fix: импортировать общий `get_tools`. small
- [ ] L13. Непоследовательная обработка ошибок в `metrics/agent.py:19-36,57-78` (`add_response`/`get_discussion_metrics` без try/except PyMongoError). small
- [ ] L14. `services/trainer/test.py` — ручной харнесс с хардкод-UUID, `cuda:0`, `print`. Fix: в `scripts/`, параметризовать, logger. small

---

## Progress

- Critical: 0/1
- High: 0/13
- Medium: 0/13
- Low: 0/14
