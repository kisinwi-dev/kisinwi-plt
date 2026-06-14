# datasets

Сервис управления датасетами платформы KiSinWi: загрузка, версионирование, валидация
и статистика датасетов классификации изображений.
**FastAPI**, порт **6500**, хранение - **файловая система** (тома в `db/datasets_storage/`).
Зависимости - через **uv** (`uv sync`, `uv run ...`).

> Это документация для разработчика: назначение, запуск, конфиг, HTTP API, устройство
> файлового хранилища, валидация и неочевидные детали поведения. Карта модулей `app/` - в конце.

## Назначение и место в пайплайне

datasets - первый сервис в потоке: он хранит сырые данные и считает по ним статистику,
которой дальше пользуются agents и trainer.

```
[ datasets ] → agents → tasker → trainer → metrics + ml_models → agent_history
```

- **frontend** загружает архивы (`POST /upload`), создаёт датасеты/версии, читает статистику;
- **agents** читают распределение классов, баланс, размеры и integrity версии для анализа;
- **trainer** забирает физические файлы версии (`{dataset_id}/{version_id}/...`) для обучения.

В отличие от tasker/trainer у datasets **нет БД и нет воркера**: это HTTP-сервис над локальной
файловой системой. Единственный фоновый процесс - очистка временной папки по TTL (см. ниже).

## Архитектура

`main.py` поднимает один uvicorn-сервер. Бизнес-логика - в `DatasetManager`
([app/core/services/dataset.py](app/core/services/dataset.py)). В отличие от синглтонов
в других сервисах, `DatasetManager` создаётся **заново на каждый запрос**
([app/api/deps.py](app/api/deps.py)) - состояния между запросами он не держит, всё лежит на диске.

Слои:

```
routers ─▶ DatasetManager ─┬─▶ core/services/validation  (структура, статистика, хеши)
(api)        (core)        ├─▶ core/services/comparison   (дельты, JS divergence, PSI, file diff)
                           ├─▶ core/services/integrity    (дубликаты, leakage по SHA256)
                           └─▶ core/filesystem            (FileSystemManager / ArchiveManager)
```

Поток создания данных:

```
POST /upload ─▶ ArchiveManager (распаковка в temp/) ─▶ POST /datasets/new или .../versions/new
                                                              │
                                          валидация структуры + статистика + SHA256
                                                              │
                                          перенос (move) из temp/ в {dataset_id}/{version_id}/
                                                              │
                                          запись {version_id}.hashes.json + metadata_ds.json
```

## Запуск

```bash
uv sync
python main.py          # uvicorn на 0.0.0.0:6500, reload=True
# либо: uv run uvicorn main:app --host 0.0.0.0 --port 6500
```

Порт в `main.py` захардкожен (`6500`); `DATASETS_SERVICE_PORT` читается только в Docker CMD.
Рабочая директория важна: `FileSystemManager` берёт корень как `<cwd>/datasets`
([app/core/filesystem/fsm.py](app/core/filesystem/fsm.py)) - запускать сервис нужно из каталога
сервиса, где этот `datasets/` существует.

### Docker

```dockerfile
FROM python:3.13-slim
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${DATASETS_SERVICE_PORT:-6500}"]
```

## Переменные окружения

Читаются в [main.py](main.py).

| Переменная                | Назначение                                    | По умолчанию |
|---------------------------|-----------------------------------------------|--------------|
| `DATASETS_TEMP_TTL_HOURS` | TTL временных загрузок в `temp/` (часы)        | `24`         |
| `DATASETS_SERVICE_PORT`   | порт сервиса (**только в Docker CMD**)         | `6500`       |

## HTTP API

Роутеры: [app/api/routers/](app/api/routers/) - `upload.py`, `datasets.py`, `versions.py`, `info.py`.

### Upload

| Метод | Путь      | Назначение |
|-------|-----------|------------|
| POST  | `/upload` | загрузить архив (`id_data` форма + `file`); распаковать в `temp/{id_data}/`. Форматы: `.zip`, `.tar`, `.tar.gz`, `.tgz` |

`id_data` связывает загрузку с будущим датасетом/версией. Повторная загрузка с тем же `id_data`
**перетирает** прежнюю temp-папку ([af.py:150-156](app/core/filesystem/af.py#L150)).

### Datasets

| Метод  | Путь                                  | Назначение |
|--------|---------------------------------------|------------|
| GET    | `/datasets/`                          | список датасетов; `?limit=&offset=&search=` (поиск по подстроке в id/name); общее число в заголовке `X-Total-Count` |
| GET    | `/datasets/{dataset_id}`              | метаданные датасета |
| POST   | `/datasets/new`                       | создать датасет из загруженных данных (тело `NewDataset`, в нём `version.id_data`) |
| PATCH  | `/datasets/{dataset_id}`              | изменить `name`/`description` (id и папка не меняются) |
| POST   | `/datasets/{dataset_id}/default_version` | сменить стандартную версию (`?default_version=`) |
| DELETE | `/datasets/{dataset_id}`              | удалить датасет |

### Versions

Префикс: `/datasets/{dataset_id}/versions`.

| Метод  | Путь                       | Назначение |
|--------|----------------------------|------------|
| GET    | `/`                        | список версий датасета |
| GET    | `/{version_id}`            | метаданные версии |
| POST   | `/new`                     | создать версию из загруженных данных (тело `NewVersion` c `id_data`) |
| PATCH  | `/{version_id}`            | изменить `name`/`description` версии |
| DELETE | `/{version_id}`            | удалить версию (нельзя удалить **стандартную** - 409) |

### Version Stats

| Метод | Путь                                        | Назначение |
|-------|---------------------------------------------|------------|
| GET   | `/{version_id}/splits`                       | полная сводка по сплитам (counts + balance + distribution + size) |
| GET   | `/{version_id}/splits/count`                 | количество изображений по сплитам |
| GET   | `/{version_id}/splits/balance`               | коэффициент баланса классов по сплитам |
| GET   | `/{version_id}/splits/distribution`          | распределение по классам в каждом сплите |
| GET   | `/{version_id}/splits/size-stats`            | статистика размеров изображений (`WxH`) |
| GET   | `/{version_id}/files`                        | список файлов версии (`?split=&limit=&offset=`), `total` в ответе |
| GET   | `/{version_id}/integrity`                    | отчёт о дубликатах и leakage (по SHA256) |

Статистика **не считается на лету**: она вычисляется один раз при создании версии и хранится
в `metadata_ds.json`. Эндпоинты `/splits/*` читают готовые числа. Исключения - `/files`
(сканирует папку версии) и `/integrity` (читает файл хешей).

### Version Compare

| Метод | Путь                            | Назначение |
|-------|---------------------------------|------------|
| GET   | `/compare`                       | полная сводка сравнения двух версий |
| GET   | `/compare/counts`                | дельты количества по сплитам/классам |
| GET   | `/compare/distribution`          | состав классов + drift-метрики (JS divergence, PSI) |
| GET   | `/compare/balance`               | изменение баланса классов |
| GET   | `/compare/size-stats`            | изменение форматов и размеров |
| GET   | `/compare/files`                 | по-файловый diff (added/removed/common) |

Версии задаются query-параметрами **`from`** и **`to`** (не `from_version`/`to_version`):
`GET /compare?from=<id>&to=<id>`. Сравнение версии самой с собой - 400.

### Info

| Метод | Путь            | Назначение |
|-------|-----------------|------------|
| GET   | `/info/health`  | статус сервиса |

Health сейчас **всегда `healthy`**: проверяется только запись `file_system`, захардкоженная как
здоровая ([app/core/health.py](app/core/health.py)) - реальная доступность диска не проверяется.

## Файловое хранилище

Корень - `<cwd>/datasets/`. Структура:

```
datasets/
├── temp/
│   └── {id_data}/                 загруженные и распакованные данные (до /new)
└── {dataset_id}/
    ├── metadata_ds.json           метаданные датасета + всех его версий
    ├── {version_id}.hashes.json   карта SHA256 файлов версии (СОСЕД папки, не внутри неё)
    └── {version_id}/
        ├── train/{class}/*.jpg
        ├── val/{class}/*.jpg
        └── test/{class}/*.jpg
```

Ожидаемая структура архива: на верхнем уровне **ровно** папки `train`, `val`, `test`, в каждой -
одинаковый набор папок-классов с изображениями. Поддерживаемые форматы изображений:
**`.jpg`, `.png`, `.jpeg`** ([fsm.py:12](app/core/filesystem/fsm.py#L12)). Служебные файлы
(`thumbs.db`, `desktop.ini`, всё начинающееся с `.`) молча игнорируются; **любой другой**
не-картиночный файл валит валидацию.

## Валидация и статистика

Валидаторы выбираются по паре `(type, task)` из реестра
([app/core/services/validation/registry.py](app/core/services/validation/registry.py)).
Поддерживается **только** `("image", "classification")`; прочие комбинации - `UnsupportedDatasetError` (400).

При создании ([image_classification.py](app/core/services/validation/image_classification.py)):

1. **Эталон классов.** Для нового датасета список классов берётся из папок в `train/` и
   фиксируется в `classes_to_idx` (индексы по алфавиту, независимо от порядка папок). Для новой
   версии классы должны **совпадать** с эталоном датасета - иначе 400.
2. **Структура.** Сверяются `train/val/test` и набор классов в каждом сплите; пустой класс - ошибка.
3. **Статистика.** По каждому изображению считаются размер (`"WxH"`), цветовой режим (PIL `img.mode`:
   RGB, L, RGBA, ...) и формат; счётчики суммируются по классам и сплитам.
4. **Целостность файлов.** Каждое изображение открывается через `PIL.Image.verify()`; **все**
   повреждённые собираются и в конце выбрасываются одной ошибкой (400) со списком путей.
5. **Хеши.** Считается SHA256 каждого изображения (относительный путь → хеш), сохраняется в
   `{version_id}.hashes.json`.

## Сравнение версий и integrity

**Drift-метрики** ([app/core/services/comparison.py](app/core/services/comparison.py)):
JS divergence (log base 2, диапазон 0-1) и PSI. Интерпретация - общий порог `(0.1, 0.25)`:
ниже `0.1` - `none`, между - `moderate`, выше `0.25` - `significant`.

**Integrity** ([app/core/services/integrity.py](app/core/services/integrity.py)) строится по карте
хешей:

- **дубликаты** - одинаковый SHA256 **внутри одного** сплита;
- **leakage** - одинаковый SHA256 **между** сплитами (`train_val`, `train_test`, `val_test`).

## Взаимодействие с сервисами

datasets - источник данных: исходящих вызовов к другим сервисам у него нет, только локальная
файловая система. Все вызовы входящие:

| Источник | Метод / путь | Когда |
|----------|--------------|-------|
| frontend | `POST /upload`, `POST /datasets/new`, `.../versions/new`, CRUD | загрузка архивов, создание датасетов/версий, чтение статистики |
| agents   | `GET /{version_id}/splits/*`, `/integrity`, `/compare/*` | анализ распределения классов, баланса, размеров, integrity и дрифта версий |
| trainer  | физические файлы версии (`{dataset_id}/{version_id}/...`) с диска | чтение данных для обучения (`ImageFolder`) |

`GET /info/health` сейчас всегда `healthy` (запись `file_system` захардкожена, см. ниже).

## Нюансы

- **Хеши - сосед папки версии** (`{version_id}.hashes.json`), а не внутри `{version_id}/`. Сделано
  специально, чтобы файл хешей не попадал в `/compare/files`.
- **Integrity недоступен для старых версий** без посчитанных хешей -
  `IntegrityReportNotAvailableError` (404, [version.py](app/core/exception/version.py#L23)).
- **temp/ чистится фоном** раз в час по `DATASETS_TEMP_TTL_HOURS` (`_temp_cleanup_loop` в
  [main.py](main.py#L18), запуск в `lifespan`).
- **Архив "разворачивается".** Если внутри одна корневая папка (заархивировали каталог целиком),
  её содержимое поднимается на уровень выше; `__MACOSX` удаляется ([af.py](app/core/filesystem/af.py#L134)).
  Защита от zip slip / path traversal встроена в распаковку.
- **Перенос - это `move`.** При создании датасета/версии данные физически перемещаются из `temp/`
  в папку датасета, temp-папка `id_data` удаляется.
- **Откат с возвратом в temp.** Если запись метаданных/хешей упала, данные **возвращаются в temp/**
  ([dataset.py:475](app/core/services/dataset.py#L475)) - повтор `/new` возможен без перезаливки архива.
- **Метаданные пишутся атомарно** через `.tmp` + `replace` ([dataset.py:381](app/core/services/dataset.py#L381)).
- **Датасет = папка с `metadata_ds.json`.** Папки без этого файла не считаются датасетами; сервис
  пока рассчитан на одного пользователя (комментарий в `get_datasets_id`).
- **Нельзя удалить стандартную версию** (`CannotDeleteDefaultVersion`, 409).
- **CORS открыт на все origins.**

## Отладка

- **Логи**: `app/logs/`, см. [app/logs/config.py](app/logs/config.py).
- **`/upload` отдаёт 500** - битый или неподдерживаемый архив; temp-папка `id_data` подчищается.
- **`/datasets/new` отдаёт 400** - ошибка валидации (структура, классы, битые изображения),
  загруженные данные удаляются из temp; при 500 (сбой записи) данные остаются в temp для повтора.
- **`/integrity` отдаёт 404** - у версии нет файла хешей (создана до появления функции).
- **Пустой список датасетов** при наличии папок - проверь, что в каждой лежит `metadata_ds.json`.

## Структура `app/`

```
app/
├── api/
│   ├── routers/      upload.py, datasets.py, versions.py, info.py
│   ├── schemas/      dataset.py, dataset_new.py (+ Update), splits.py, comparison.py,
│   │                 integrity.py, files.py, info.py
│   └── deps.py       get_dataset_manager (DatasetManager на каждый запрос)
├── core/
│   ├── services/
│   │   ├── dataset.py        DatasetManager - вся бизнес-логика (CRUD, promote, rollback, paths)
│   │   ├── comparison.py     дельты, JS divergence, PSI, file diff
│   │   ├── integrity.py      дубликаты и leakage по SHA256
│   │   └── validation/       registry.py (реестр (type,task)) + image_classification.py
│   ├── filesystem/
│   │   ├── fsm.py            FileSystemManager (навигация, хеши, размер, фильтр изображений)
│   │   └── af.py             ArchiveManager (распаковка, flatten, zip slip, TTL-очистка temp)
│   ├── exception/   dataset.py, version.py, base.py (CoreException)
│   └── health.py    check_health_all (file_system)
└── logs/            логгер
```
