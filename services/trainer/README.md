# trainer

Сервис обучения DL-моделей (классификация изображений) платформы KiSinWi.
**FastAPI + PyTorch** (timm, torchmetrics), порт **6200**.
Зависимости - через **pip / `requirements.txt`** (КРОМЕ ФРЕЙМВОРКА TORCH).

> Это документация для разработчика: назначение, запуск, конфиг, поток обучения,
> взаимодействие с сервисами и отладка. Карта модулей `app/` - в конце файла.

## Назначение и место в пайплайне

trainer непосредственно обучает модель и выгружает результат:

```
datasets → agents → tasker → [ trainer ] → metrics + ml_models → agent_history
```

- читает данные датасета с локального диска (`datasets/{dataset_id}/{version_id}/...`);
- берёт `train_params` обучаемой модели из **ml_models**;
- забирает задачи из **tasker** и репортит туда прогресс/статус;
- шлёт метрики и статус обучения в **metrics**;
- выгружает обученную модель (ONNX) обратно в **ml_models**.

## Архитектура

Воркер + справочный HTTP API. `main.py` через `asyncio.gather` поднимает **параллельно** две вещи:

1. **uvicorn-сервер** - справочный HTTP API (только справка и валидация конфига).
2. **воркер** `to_work()` ([app/core/worker.py](app/core/worker.py)) - единственный в системе,
   в цикле (раз в 1 c) опрашивает tasker `GET /tasks/next`.

**Обучение через HTTP не запускается** - его запускает воркер, забирая задачу из tasker.

```
                       ┌─────────── trainer (main.py, asyncio.gather) ───────────┐
                       │                                                          │
   agents/frontend ──▶ │  uvicorn API  (GET /info/*, POST /config/validate)       │
                       │                                                          │
   tasker  ◀───────────│  worker to_work():  GET /tasks/next → training_model()   │
   ml_models ◀─────────│       (train_params)            └─▶ Trainer.train()      │
   metrics ◀───────────│       (метрики, статус)              └─▶ ONNX → ml_models │
                       └──────────────────────────────────────────────────────────┘
```

## Запуск

```bash
pip install -r requirements.txt   # torch/torchvision не в requirements - см. Docker
python main.py                    # поднимает и API, и воркер
```

Перед стартом `config_services.check_services()` ([app/config/__init__.py](app/config/__init__.py))
проверяет доступность tasker / ml_models / metrics (`GET /info/health`).

`test.py` - ручной прогон `training_model` без tasker (для локальной отладки обучения).

### Docker

torch уже есть в базовом образе, в `requirements.txt` его нет; torchvision ставится отдельно
под CUDA 12.8 ([Dockerfile](Dockerfile)):

```dockerfile
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN pip install --no-cache-dir --break-system-packages torchvision \
    --index-url https://download.pytorch.org/whl/cu128
CMD ["python", "main.py"]
```

## Переменные окружения

Загружаются через `dotenv` из `.env`. Источник - [app/config/__init__.py](app/config/__init__.py).

| Переменная                 | Назначение                                  | По умолчанию       |
|----------------------------|---------------------------------------------|--------------------|
| `HF_TOKEN`                 | токен Hugging Face для pretrained-весов timm (нужен для gated-моделей) | - (warning в логах) |
| `HF_HOME`                  | кэш весов huggingface_hub (в Docker - bind-mount `./db/hf_cache`) | `~/.cache/huggingface` |
| `TORCH_HOME`               | кэш весов torch.hub (legacy-модели timm)    | `~/.cache/torch`   |
| `HF_HUB_DOWNLOAD_TIMEOUT`  | таймаут (сек) на скачивание весов с HF Hub  | `30`               |
| `HF_HUB_DISABLE_XET`       | `1` - отключить Xet-бэкенд (см. «Загрузка pretrained-весов») | `1` (в compose)    |
| `TASKER_DOMAIN`            | host:port сервиса tasker                    | `localhost:6110`   |
| `ML_MODELS_DOMAIN`         | host:port сервиса ml_models                 | `localhost:6300`   |
| `METRICS_DOMAIN`           | host:port сервиса metrics                   | `localhost:6310`   |
| `TRAINER_SERVICE_PORT`     | порт HTTP API trainer                       | `6200`             |

## HTTP API

Только справка и валидация - запуска/отмены обучения здесь нет.

| Метод | Путь                     | Назначение |
|-------|--------------------------|------------|
| GET   | `/info/example_config`   | JSON-схема конфига обучения (`TaskParams.model_json_schema()`) - **динамическая схема для agents** |
| GET   | `/info/ml_models`        | список доступных моделей timm; опц. `?filter=*resnet*` |
| GET   | `/info/device`           | инфо о CPU / GPU / RAM / CUDA |
| GET   | `/info/optimizers`       | доступные оптимизаторы (`torch.optim`) |
| GET   | `/info/schedulers`       | доступные планировщики (`torch.optim.lr_scheduler`) |
| GET   | `/info/metrics`          | доступные метрики (`METRICS_REGISTRY`) |
| GET   | `/info/augmentations`    | доступные трансформации (`ALLOWED_TRANSFORMS`) |
| GET   | `/info/health`           | статус trainer + зависимых сервисов |
| POST  | `/config/validate`       | проверить конфиг без запуска → `{valid, errors}` |

Роутеры: [app/api/routers/train_settings.py](app/api/routers/train_settings.py),
[app/api/routers/validate_config.py](app/api/routers/validate_config.py).
`POST /config/validate` проверяет соответствие схеме **и** существование модели / loss /
optimizer / scheduler / метрик / трансформаций + доступность устройства на этом инстансе
([app/core/utils/validate_config.py](app/core/utils/validate_config.py)).

## Конфиг обучения (`TaskParams`)

Структура **динамическая** - не хардкодить, брать из `GET /info/example_config`.
Схема: [app/api/schemas/tasker.py](app/api/schemas/tasker.py) (+ `trainer.py`, `data.py`, `ml_models.py`).

```jsonc
{
  "data_loader_params": {            // app/api/schemas/data.py
    "dataset_id": "dataset_123",
    "version_id": "v1.0",
    "batch_size": 32,
    "num_workers": 2,
    "train_transforms_config": [ {"name": "RandomResizedCrop", "params": {"size": [224,224]}} ],
    "val_and_test_transforms_config": [ {"name": "Resize", "params": {"size": [224,224]}} ]
    // img_w_size / img_h_size - deprecated, размер задаётся трансформациями
  },
  "model_params": {                  // app/api/schemas/ml_models.py
    "type": "resnet50",              // имя модели из timm
    "pretrained": true
  },
  "trainer_params": {                // app/api/schemas/trainer.py
    "loss_fn":   {"name": "CrossEntropyLoss", "params": {"reduction": "mean", "label_smoothing": 0.1}},
    "optimizer": {"name": "AdamW",            "params": {"lr": 0.001, "weight_decay": 0.01}},
    "scheduler": {"name": "CosineAnnealingLR","params": {"T_max": 50, "eta_min": 1e-6}},
    "epochs": 50,
    "early_stop": {"metric_name": "loss", "patience": 4, "min_delta": 0.001, "mode": "min"},
    "use_amp": false,                // Automatic Mixed Precision - только GPU, на CPU игнорируется
    "grad_clip_norm": 1.0            // clip_grad_norm_; null - без ограничения
  },
  "device": "cuda:0"                 // 'cuda:N' или 'cpu'
}
```

- `loss_fn.name` - из `torch.nn`; `optimizer.name` - из `torch.optim`;
  `scheduler.name` - из `torch.optim.lr_scheduler`; `params` подставляются как `**kwargs`.
- `early_stop.metric_name` - имя метрики (или `loss`); по `val_{metric_name}` выбирается лучшая
  эпоха и проверяется ранняя остановка; `mode` - `min` (loss) или `max` (accuracy/f1...).
- **Метрики не конфигурируются**: всегда считается полный набор из `METRICS_REGISTRY` + loss на
  train/val (см. [IMPROVEMENTS.md](IMPROVEMENTS.md)).

## Поток обучения

Оркестратор - [app/core/__init__.py](app/core/__init__.py)::`training_model(config, model_id)`.
Прогресс задачи - именованные константы `PROGRESS_*`:

| Прогресс | Этап |
|----------|------|
| 0–19     | подготовка: `setup_device` → `create_dataloaders` (ImageFolder train/val/test) → **pre-download весов** (если `pretrained`, этап 6–8) → `get_model` (timm) → `MetricesClient` → сборка `Trainer` |
| 20–80    | `Trainer.train()` - цикл эпох: forward/backward, AMP (GPU), grad clipping, шаг scheduler, метрики, early stop, чекпоинт лучшей эпохи |
| -        | восстановление лучших весов из чекпоинта → прогон на test → `send_checkpoint_info` в metrics |
| 81–95    | экспорт в ONNX (`save_model_to_onnx`, классы в metadata) → upload в ml_models → удаление чекпоинта |
| 100      | задаче ставится `completed` (воркер) |

Класс `Trainer` - [app/core/trainer/train_model.py](app/core/trainer/train_model.py).
Синхронная часть обучения выполняется в `asyncio.to_thread`, чтобы воркер мог проверять отмену.

## Загрузка pretrained-весов

При `model_params.pretrained = true` веса скачиваются с Hugging Face Hub. Скачивание вынесено в
**отдельный этап pre-download** (прогресс 6–8) ДО `get_model`
([app/core/models/downloader.py](app/core/models/downloader.py)::`predownload_weights`), после чего
`timm.create_model(pretrained=True)` берёт веса из кэша мгновенно. Зачем так:

- **Видимый прогресс.** Раньше между «Загрузка модели...» и «модель загружена» была тишина, и было
  непонятно, идёт ли скачивание или всё зависло. Теперь `_LoggingTqdm` пишет в логи строки вида
  `Скачивание весов: 49% (6.1/12.3 МБ)`, а в `status_info` задачи идёт «Скачивание весов модели...».
- **Кэш на volume.** `HF_HOME` / `TORCH_HOME` указывают в bind-mount `./db/hf_cache`. Без него веса
  качались в эфемерный слой контейнера и скачивались **заново при каждом пересоздании**; теперь модель
  качается один раз и переживает рестарты.
- **Отключённый Xet (`HF_HUB_DISABLE_XET=1`).** По умолчанию `huggingface_hub` качает через Xet-бэкенд
  (`hf_xet`, ходит на `cas-server.xethub.hf.co`). В нашем окружении этот хост недоступен, и скачивание
  **виснет без обрыва** (это и был «бесконечный цикл») - при этом обычный `huggingface.co` доступен.
  Флаг форсит классический HTTP-путь, который уважает `HF_HUB_DOWNLOAD_TIMEOUT` и умеет докачку (resume)
  после read-timeout. Если на другой машине Xet работает - флаг можно убрать (Xet быстрее).
- **Fallback.** Модель не на HF Hub (legacy torch.hub) → pre-download пропускается, `get_model` качает
  её сам в `TORCH_HOME`. Сетевая ошибка/таймаут → исключение пробрасывается, задача уходит в `failed`
  (через воркер), а не висит вечно.

Тест: [tests/test_predownload.py](tests/test_predownload.py) (запуск в контейнере
`python -m tests.test_predownload`, модель переопределяется через env `HF_TEST_MODEL`).

## Жизненный цикл задачи и модели

Статус модели в ml_models: `draft → training → completed`, **откат в `draft`** при ошибке/отмене.

- **Старт**: воркер ставит задаче `running`, модели `training`, шлёт в metrics `in_progress`.
- **Успех**: задача `completed`, модель `completed`.
- **Ошибка / CUDA OOM**: задача `failed` (с retry `report_task_failed`, 3 попытки), модель → `draft`.
  При неудаче всех попыток - CRITICAL-лог, задача останется `running` (подхватится при рестарте).
- **Отмена**: tasker ставит `cancelled`; trainer ловит это на границе эпох →
  `TaskCancelledError`; статус задачи **не трогаем** (его уже выставил tasker), в metrics шлём
  `cancelled`, модель → `draft`.
- **Старт воркера**: `recover_orphaned_tasks()` помечает все «зависшие» задачи в статусе `running`
  как `failed` (воркер один - значит они осиротели после падения).

## Взаимодействие с сервисами

Исходящие вызовы trainer:

| Сервис    | Метод / путь                          | Когда |
|-----------|---------------------------------------|-------|
| tasker    | `GET /tasks/next`                     | опрос задач (раз в 1 c) |
| tasker    | `GET /tasks` (status=running)         | на старте, `recover_orphaned_tasks` |
| tasker    | `GET /tasks/{id}`                     | проверка отмены на границе эпох |
| tasker    | `POST /tasks/{id}/status`             | прогресс / статус / ошибка |
| ml_models | `GET /versions/{model_id}`            | получить `train_params` |
| ml_models | `PATCH /versions/{model_id}`          | статус модели (`training`/`completed`/`draft`) |
| ml_models | `POST /versions/{model_id}/files`     | upload ONNX (multipart, timeout 60 c) |
| metrics   | `POST /models/{id}/status`            | статус обучения (`in_progress`/`completed`/`failed`/`cancelled`) |
| metrics   | `POST /models/{id}/metrics`           | поэпоховые скаляры |
| metrics   | `POST /models/{id}/checkpoint`        | эпоха/метрика/значение сохранённых весов |
| metrics   | `POST /models/{id}/class-report`      | confusion matrix + per-class P/R/F1 (после test) |

Метрики: при обучении на **GPU** живут на CPU и копятся в фоновом потоке (дренируется в `compute`);
на **CPU** - последовательно. Поэпохово шлются только скаляры; не-скалярные тензоры
(confusion matrix, per-class) уходят отдельным class report после теста.
Подробности - в `app/service/metrices/` (`mc.py`, `collection.py`).

## Нюансы

- **Написание `metrices`** (не `metrics`) в именах модулей/схем - намеренное, не опечатка.
- **Метрики не конфигурируются**: всегда считается полный набор из `METRICS_REGISTRY` + loss на
  train/val; секции метрик в конфиге обучения нет (старый ключ `metrices_params` игнорируется).
- **AMP только на GPU**: `use_amp` на CPU игнорируется.
- **Единственный воркер**: на старте `recover_orphaned_tasks()` помечает все `running`-задачи
  `failed` (раз воркер один - они осиротели после падения).
- **Путь к датасетам относительный** (`datasets/{dataset_id}/{version_id}/...`) - запускать сервис
  нужно из корня сервиса.
- **`img_w_size`/`img_h_size`** в `DataLoaderParams` - deprecated, размер задаётся трансформациями.
- **torch не в `requirements.txt`**: он уже в базовом образе, torchvision ставится отдельно под CUDA.

## Отладка

- **Логи**: `app/logs/app_json.log`, уровень `DEBUG` ([app/logs/config.py](app/logs/config.py)).
- **Нет `HF_TOKEN`** → warning в логах, pretrained-веса timm могут не скачаться (gated-модели).
- **Скачивание весов «зависло»** → почти всегда Xet-бэкенд: проверь, что выставлен `HF_HUB_DISABLE_XET=1`
  (см. «Загрузка pretrained-весов»). В логах ищи `Скачивание весов: N%` - если их нет и висит на
  «Скачивание весов модели...», значит соединение не идёт. Кэш в `db/hf_cache` создаётся под root -
  чистить застрявшие `*.incomplete` через контейнер, не с хоста.
- **Сервис недоступен на старте** → `check_services` пометит его `unhealthy` в `/info/health`.
- **Нет папки датасета** (`datasets/{dataset_id}/{version_id}/...`) → ошибка при `create_dataloaders`;
  путь сейчас относительный - запускать из корня сервиса.
- **CUDA OOM** → задача `failed`, cache очищается; уменьшить `batch_size` / размер изображений / отключить `use_amp`.

## Структура `app/`

Краткая карта модулей:

```
app/
├── __init__.py        FastAPI app + uvicorn
├── config/            URL-ы сервисов, check_services()
├── api/
│   ├── routers/       /info/* (train_settings), /config/validate
│   └── schemas/       TaskParams, TrainerParams, DataLoaderParams, ModelParams, ...
├── core/
│   ├── __init__.py    training_model() - оркестратор
│   ├── worker.py      to_work() - воркер-цикл
│   ├── trainer/       класс Trainer (цикл эпох, AMP, early stop, чекпоинт)
│   ├── models/        get_model (timm), predownload_weights (downloader.py - кэш/прогресс весов)
│   ├── datas/         create_dataloaders, ALLOWED_TRANSFORMS
│   └── utils/         setup_device, validate_task_params, save_model_to_onnx
├── service/           HTTP-клиенты: tasker / ml_models / metrices
└── logs/              логгер
```
