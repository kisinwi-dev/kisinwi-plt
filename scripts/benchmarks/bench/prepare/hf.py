"""Подготовка HF-датасета: скачать с HuggingFace, разложить в формат платформы
(train/val/test/<class>/*.jpg), упаковать в zip и загрузить в datasets.

Точка входа - prepare_hf(key); управление через benchmark.py.
<key> - ключ source="huggingface" из config.DATASETS (beans, cifar10, ...).
"""
import io
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from bench import load_env

# .env и HF-настройки до импорта datasets: Xet-backend виснет на скачивании,
# дефолтный таймаут 10s рвёт крупные parquet.
load_env()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from datasets import load_dataset
from PIL import Image

from bench.config import DATASETS
from bench.common import (
    upload_archive, create_dataset, find_dataset_by_name,
    load_registry, save_registry, zip_folder,
)


def _class_names(ds, label_col):
    # ClassLabel даёт метки-индексы 0..n-1, иначе classes[label] не резолвится.
    feat = ds.features[label_col]
    names = getattr(feat, "names", None)
    if not names:
        raise RuntimeError(
            f"колонка '{label_col}' не ClassLabel - добавьте явный список классов"
        )
    return [str(n).replace("/", "_").replace(" ", "_") for n in names]


def _save_image(img, dst: Path):
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
    img = img.convert("RGB")
    # quality=95 + subsampling=0 (4:4:4): меньше артефактов пересжатия, важно для
    # мелких источников (CIFAR 32x32).
    img.save(dst, format="JPEG", quality=95, subsampling=0)


def _stratified_split(indices_by_class: dict, fraction: float, seed: int):
    """Делит индексы каждого класса на (main, held) по доле fraction."""
    rng = random.Random(seed)
    main, held = defaultdict(list), defaultdict(list)
    for cls, idxs in indices_by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_held = max(1, int(len(idxs) * fraction)) if len(idxs) > 1 else 0
        held[cls] = idxs[:n_held]
        main[cls] = idxs[n_held:]
    return main, held


def build_dataset_folder(key: str, out_root: Path) -> Path:
    cfg = DATASETS[key]
    image_col, label_col = cfg["image_col"], cfg["label_col"]

    print(f"[{key}] загрузка с HuggingFace: {cfg['hf_id']}")
    hf_splits = {}
    needed = set(cfg["split_map"].values())
    for hf_split in needed:
        if hf_split == "@from_train":
            continue
        hf_splits[hf_split] = load_dataset(cfg["hf_id"], split=hf_split)

    train_hf_name = cfg["split_map"]["train"]
    classes = _class_names(hf_splits[train_hf_name], label_col)
    print(f"[{key}] классов: {len(classes)}")

    # индексы train по классам - для выделения val/test
    def indices_by_class(ds):
        by = defaultdict(list)
        for i, lab in enumerate(ds[label_col]):
            by[lab].append(i)
        return by

    # источник для каждого целевого сплита: plan[split] = (hf_dataset, indices)
    plan = {}
    train_ds = hf_splits[train_hf_name]
    train_idx_by_cls = indices_by_class(train_ds)

    from_train_splits = [s for s, v in cfg["split_map"].items() if v == "@from_train"]
    remaining = train_idx_by_cls
    for split in from_train_splits:
        main, held = _stratified_split(remaining, cfg["val_fraction"], seed=42)
        plan[split] = (train_ds, held)
        remaining = main
    plan["train"] = (train_ds, remaining)

    for split, hf_name in cfg["split_map"].items():
        if hf_name == "@from_train":
            continue
        if split == "train":
            continue  # train уже учтён через remaining
        ds = hf_splits[hf_name]
        plan[split] = (ds, indices_by_class(ds))

    # max_per_class и раскладка по папкам
    max_pc = cfg.get("max_per_class")
    rng = random.Random(123)

    out_dir = out_root / key
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split, (ds, idx_by_cls) in plan.items():
        for cls_id, idxs in idx_by_cls.items():
            idxs = list(idxs)
            if max_pc and len(idxs) > max_pc:
                rng.shuffle(idxs)
                idxs = idxs[:max_pc]
            for n, i in enumerate(idxs):
                cls_name = classes[cls_id]
                dst_dir = out_dir / split / cls_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                _save_image(ds[i][image_col], dst_dir / f"{split}_{cls_id}_{n}.jpg")
        print(f"[{key}] {split}: разложено")

    return out_dir


def upload_to_platform(key: str, zip_path: Path):
    cfg = DATASETS[key]
    id_data = key
    # имя с ключом: id платформа выдаёт как UUID, ищем датасет по имени
    platform_name = f"{cfg['name']} [{key}]"

    reg = load_registry()
    if key in reg and find_dataset_by_name(platform_name):
        print(f"[{key}] уже зарегистрирован ({reg[key]['dataset_id']}) - пропуск")
        return reg[key]

    print(f"[{key}] upload архива...")
    upload_archive(id_data, str(zip_path))

    payload = {
        "name": platform_name,
        "description": cfg["role"],
        "type": "image",
        "task": "classification",
        "version": {
            "id_data": id_data,
            "name": "v1",
            "description": "benchmark prepared from HuggingFace",
            "sources": [{"type": "huggingface", "url": cfg["url"]}],
        },
    }
    print(f"[{key}] создание датасета...")
    create_dataset(payload)

    info = find_dataset_by_name(platform_name)
    if info is None:
        raise RuntimeError(f"датасет '{platform_name}' создан, но не найден в списке")
    leakage = info["versions"][0].get("integrity", {}).get("leakage", {})
    print(f"[{key}] dataset_id={info['id']} классов={info['classes_count']} leakage={leakage}")

    reg[key] = {
        "dataset_id": info["id"],
        "version_id": info["default_version_id"],
        "name": platform_name,
        "num_classes": info["classes_count"],
    }
    save_registry(reg)
    return reg[key]


def prepare_hf(key: str, workdir: str = "/tmp/kisinwi_bench",
               upload: bool = True) -> dict | None:
    """Подготовить HF-датасет и (опц.) загрузить в платформу.

    Возвращает запись реестра (dataset_id/version_id/...) при upload, иначе None.
    """
    out_root = Path(workdir)
    out_root.mkdir(parents=True, exist_ok=True)

    folder = build_dataset_folder(key, out_root)
    zip_path = zip_folder(folder)
    entry = upload_to_platform(key, zip_path) if upload else None
    # распакованные изображения удаляем, zip оставляем
    shutil.rmtree(folder, ignore_errors=True)
    print(f"[{key}] готово")
    return entry
