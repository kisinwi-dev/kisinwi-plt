"""Конфигурация benchmark-прогона: адреса сервисов и описание датасетов.

Поле `source` выбирает prepare-модуль: "huggingface" (bench.prepare.hf) или
"kaggle" (bench.prepare.kaggle).
"""
import os

# --- адреса сервисов платформы ---
def _url(env_name: str, default: str) -> str:
    return os.getenv(env_name, default).rstrip("/")


DATASETS_URL = _url("BENCH_DATASETS_URL", "http://localhost:6500")
AGENTS_URL = _url("BENCH_AGENTS_URL", "http://localhost:6400")
TASKER_URL = _url("BENCH_TASKER_URL", "http://localhost:6110")
AGENT_HISTORY_URL = _url("BENCH_AGENT_HISTORY_URL", "http://localhost:6410")
FRONTEND_URL = _url("BENCH_FRONTEND_URL", "http://localhost:6001")

# --- описание датасетов ---
# Общие поля: source, name, baseline, role, url.
# HF-поля:
#   hf_id        - путь датасета на HuggingFace Hub
#   image_col    - колонка с изображением
#   label_col    - колонка с меткой класса (ClassLabel)
#   split_map    - сплиты платформы (train/val/test) -> сплиты HF;
#                  "@from_train" - выделить из train стратифицированно
#   val_fraction - доля train в val при "@from_train"
#   baseline     - ожидаемая test accuracy (ResNet50 transfer)
# Kaggle-поля: kaggle_refs, platform_name, num_classes.

KAGGLE_REFS = [
    "xhlulu/140k-real-and-fake-faces",
    "manjilkarki/deepfake-and-real-images",
    "prithivsakthiur/deepfake-vs-real-20k",
    "ciplab/real-and-fake-face-detection",
    "sachchitkunichetty/rvf10k",
    "maysuni/wild-deepfake",
]

DATASETS = {
    "beans": {
        "source": "huggingface",
        "name": "Beans",
        "hf_id": "AI-Lab-Makerere/beans",
        "image_col": "image",
        "label_col": "labels",
        "split_map": {"train": "train", "val": "validation", "test": "test"},
        "baseline": 0.98,
        "url": "https://huggingface.co/datasets/AI-Lab-Makerere/beans",
        "role": "sanity, быстрый прогон (3 класса болезней листьев)",
    },
    "oxford_pets": {
        "source": "huggingface",
        "name": "Oxford-IIIT Pets",
        "hf_id": "timm/oxford-iiit-pet",
        "image_col": "image",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "test"},
        "val_fraction": 0.1,
        "baseline": 0.93,
        "url": "https://huggingface.co/datasets/timm/oxford-iiit-pet",
        "role": "fine-grained (37 пород кошек/собак)",
    },
    "cifar10": {
        "source": "huggingface",
        "name": "CIFAR-10",
        "hf_id": "uoft-cs/cifar10",
        "image_col": "img",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "test"},
        "val_fraction": 0.1,
        "baseline": 0.96,
        "url": "https://huggingface.co/datasets/uoft-cs/cifar10",
        "role": "классика, мелкие изображения 32x32 (10 классов)",
    },
    "flowers102": {
        "source": "huggingface",
        "name": "Oxford Flowers-102",
        "hf_id": "nelorth/oxford-flowers",
        "image_col": "image",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "test"},
        "val_fraction": 0.1,
        "baseline": 0.96,
        "url": "https://huggingface.co/datasets/nelorth/oxford-flowers",
        "role": "много классов (102 вида цветов)",
    },
    "food101": {
        "source": "huggingface",
        "name": "Food-101 (subset)",
        "hf_id": "ethz/food101",
        "image_col": "image",
        "label_col": "label",
        "split_map": {"train": "train", "val": "@from_train", "test": "validation"},
        "val_fraction": 0.1,
        "max_per_class": 100,
        "baseline": 0.80,
        "url": "https://huggingface.co/datasets/ethz/food101",
        "role": "сложный, нагрузочный (101 класс еды, сабсэмпл 100/класс)",
    },
    "master_kaggle": {
        "source": "kaggle",
        "name": "Deepfake Real vs Fake",
        "baseline": 0.99,   # ~99.3% на test у референс-ноутбука (~87.5% на unseen);
                            # наш split 80/10/10 повторяет распределения автора.
        "num_classes": 2,
        "platform_name": "Deepfake Real vs Fake [master_kaggle-benchmark]",
        "role": "Пул 6 Kaggle-датасетов real vs deepfake (master_kaggle benchmark)",
        "kaggle_refs": KAGGLE_REFS,
        "url": "https://www.kaggle.com/code/muqaddasejaz/deepfake-image-detection",
    },
}

# Датасеты по умолчанию для полного benchmark. master_kaggle не входит: тяжёлый,
# требует надзора, запускается по имени - `benchmark.py master_kaggle`.
DEFAULT_SUITE = ["beans", "cifar10", "oxford_pets", "flowers102", "food101"]
