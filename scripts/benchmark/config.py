"""Конфигурация benchmark-прогона: адреса сервисов и описание датасетов.

Каждый датасет качается с HuggingFace и конвертируется в формат платформы
(train/val/test/<class>/*.jpg). baseline — типичное published-значение test
accuracy для ResNet50 с ImageNet-предобучением (transfer learning); именно
такой режим ожидается от платформы.
"""
import os

# --- адреса сервисов платформы ---
# Дефолт — localhost; переопределяется env (напр. docker-hostname или удалённый стенд):
#   BENCH_DATASETS_URL, BENCH_AGENTS_URL, BENCH_TASKER_URL, BENCH_AGENT_HISTORY_URL.
def _url(env_name: str, default: str) -> str:
    return os.getenv(env_name, default).rstrip("/")


DATASETS_URL = _url("BENCH_DATASETS_URL", "http://localhost:6500")
AGENTS_URL = _url("BENCH_AGENTS_URL", "http://localhost:6400")
TASKER_URL = _url("BENCH_TASKER_URL", "http://localhost:6110")
AGENT_HISTORY_URL = _url("BENCH_AGENT_HISTORY_URL", "http://localhost:6410")
# Фронтенд: в отчёт кладём deep-link'и на сравнение моделей (интерпретация «почему»).
FRONTEND_URL = _url("BENCH_FRONTEND_URL", "http://localhost:6001")

# Итераций пайплайна по умолчанию (recovery при неудачном выборе архитектуры).
# Единый источник для benchmark.py и run_benchmark.py.
DEFAULT_MAX_ITER = 3

# --- описание датасетов ---
# Поля:
#   hf_id        — путь датасета на HuggingFace Hub
#   hf_config    — config name (если требуется), иначе None
#   image_col    — колонка с изображением
#   label_col    — колонка с меткой класса (ClassLabel)
#   split_map    — соответствие сплитов платформы (train/val/test) сплитам HF.
#                  Спецзначения: "@from_train" — выделить из train стратифицированно.
#   val_fraction — доля train, уходящая в val, если val == "@from_train"
#   max_per_class— ограничение числа изображений на класс на сплит (None = всё).
#                  Нужно для тяжёлых датасетов, чтобы in-memory распаковка на
#                  сервере datasets и время обучения оставались вменяемыми.
#   baseline     — ожидаемая test accuracy (ResNet50 transfer), доля [0..1]
#   subsampled   — True, если max_per_class режет датасет и baseline ориентировочный

DATASETS = {
    "beans": {
        "name": "Beans",
        "hf_id": "AI-Lab-Makerere/beans",
        "hf_config": None,
        "image_col": "image",
        "label_col": "labels",
        "split_map": {"train": "train", "val": "validation", "test": "test"},
        "val_fraction": 0.1,
        "max_per_class": None,
        "baseline": 0.98,
        "subsampled": False,
        "url": "https://huggingface.co/datasets/AI-Lab-Makerere/beans",
        "role": "sanity, быстрый прогон (3 класса болезней листьев)",
    },
    "imagenette": {
        "name": "Imagenette",
        "hf_id": "frgfm/imagenette",
        "hf_config": "320px",
        "image_col": "image",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "validation"},
        "val_fraction": 0.1,
        "max_per_class": None,
        "baseline": 0.99,
        "subsampled": False,
        "url": "https://huggingface.co/datasets/frgfm/imagenette",
        "role": "лёгкий, крупные реальные изображения (10 классов ImageNet)",
        # ВНИМАНИЕ: не поддерживается текущей версией datasets (устаревший
        # dataset-скрипт источника). Исключён из DEFAULT_SUITE; поштучный запуск
        # `benchmark.py imagenette` упадёт на load_dataset. Запись оставлена как ссылка.
    },
    "oxford_pets": {
        "name": "Oxford-IIIT Pets",
        "hf_id": "timm/oxford-iiit-pet",
        "hf_config": None,
        "image_col": "image",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "test"},
        "val_fraction": 0.1,
        "max_per_class": None,
        "baseline": 0.93,
        "subsampled": False,
        "url": "https://huggingface.co/datasets/timm/oxford-iiit-pet",
        "role": "fine-grained (37 пород кошек/собак)",
    },
    "cifar10": {
        "name": "CIFAR-10",
        "hf_id": "uoft-cs/cifar10",
        "hf_config": None,
        "image_col": "img",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "test"},
        "val_fraction": 0.1,
        "max_per_class": None,
        "baseline": 0.96,
        "subsampled": False,
        "url": "https://huggingface.co/datasets/uoft-cs/cifar10",
        "role": "классика, мелкие изображения 32x32 (10 классов)",
    },
    "flowers102": {
        "name": "Oxford Flowers-102",
        "hf_id": "nelorth/oxford-flowers",
        "hf_config": None,
        "image_col": "image",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "test"},
        "val_fraction": 0.1,
        "max_per_class": None,
        "baseline": 0.96,
        "subsampled": False,
        "url": "https://huggingface.co/datasets/nelorth/oxford-flowers",
        "role": "много классов (102 вида цветов)",
    },
    "food101": {
        "name": "Food-101 (subset)",
        "hf_id": "ethz/food101",
        "hf_config": None,
        "image_col": "image",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "validation"},
        "val_fraction": 0.1,
        "max_per_class": 100,
        "baseline": 0.80,
        "subsampled": True,
        "url": "https://huggingface.co/datasets/ethz/food101",
        "role": "сложный, нагрузочный (101 класс еды, сабсэмпл 100/класс)",
    },
}

# Датасеты по умолчанию для полного benchmark.
# imagenette исключён: источник frgfm/imagenette использует устаревший
# dataset-скрипт, не поддерживаемый новой версией datasets. Роль «лёгкий
# датасет с крупными реальными изображениями» закрывает food101 (subset).
DEFAULT_SUITE = ["beans", "cifar10", "oxford_pets", "flowers102", "food101"]
