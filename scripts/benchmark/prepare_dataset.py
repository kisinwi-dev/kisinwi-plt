"""Подготовка benchmark-датасета: скачать с HuggingFace, разложить в формат
платформы (train/val/test/<class>/*.jpg), упаковать в zip и загрузить в datasets.

Запуск:
    python prepare_dataset.py <key> [--workdir DIR] [--no-upload]

<key> — ключ из config.DATASETS (beans, imagenette, ...).
"""
import argparse
import io
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

def _load_env():
    """Подхватить .env рядом со скриптом и настроить HF до импорта datasets.

    HF_TOKEN huggingface_hub читает из окружения сам — достаточно его выставить.
    Xet-backend виснет на скачивании, поэтому отключаем; таймаут поднимаем,
    чтобы крупные parquet не рвались на дефолтных 10s.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


_load_env()

from datasets import load_dataset
from PIL import Image

from config import DATASETS
from common import (
    upload_archive, create_dataset, find_dataset_by_name,
    load_registry, save_registry,
)


def _class_names(ds, label_col):
    # Требуем ClassLabel: тогда метки — индексы 0..n-1, и classes[label] корректно
    # резолвит имя класса в build_dataset_folder. Произвольные метки сюда не лезут.
    feat = ds.features[label_col]
    names = getattr(feat, "names", None)
    if not names:
        raise RuntimeError(
            f"колонка '{label_col}' не ClassLabel — добавьте явный список классов"
        )
    # безопасные имена папок
    return [str(n).replace("/", "_").replace(" ", "_") for n in names]


def _save_image(img, dst: Path):
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
    img = img.convert("RGB")
    # quality=95 + subsampling=0 (4:4:4) минимизируют артефакты повторного сжатия,
    # критично для уже мелких источников (напр. CIFAR 32x32), где лоссовый JPEG
    # заметно бьёт по accuracy. Расширение остаётся .jpg для совместимости.
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
    cfg_name = cfg.get("hf_config")

    print(f"[{key}] загрузка с HuggingFace: {cfg['hf_id']}"
          + (f" ({cfg_name})" if cfg_name else ""))
    hf_splits = {}
    needed = set(cfg["split_map"].values())
    for hf_split in needed:
        if hf_split == "@from_train":
            continue
        hf_splits[hf_split] = load_dataset(cfg["hf_id"], cfg_name, split=hf_split)

    train_hf_name = cfg["split_map"]["train"]
    classes = _class_names(hf_splits[train_hf_name], label_col)
    print(f"[{key}] классов: {len(classes)}")

    # Индексы train по классам — нужны для выделения val/test из train
    def indices_by_class(ds):
        by = defaultdict(list)
        for i, lab in enumerate(ds[label_col]):
            by[lab].append(i)
        return by

    # Резолвим источник для каждого целевого сплита платформы
    # plan[split] = (hf_dataset, list_of_indices)
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

    # Если у train не было @from_train сплитов, plan["train"] = remaining (полный)
    # Применяем max_per_class и раскладываем
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


def zip_folder(folder: Path) -> Path:
    zip_path = folder.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(folder):
            for fn in files:
                fp = Path(root) / fn
                zf.write(fp, fp.relative_to(folder))
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[zip] {zip_path.name}: {size_mb:.1f} MB")
    return zip_path


def upload_to_platform(key: str, zip_path: Path):
    cfg = DATASETS[key]
    id_data = key
    # Уникальное имя с ключом — платформа выдаёт UUID, находим датасет по имени.
    platform_name = f"{cfg['name']} [{key}]"

    reg = load_registry()
    if key in reg and find_dataset_by_name(platform_name):
        print(f"[{key}] уже зарегистрирован ({reg[key]['dataset_id']}) — пропуск")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(DATASETS.keys()))
    ap.add_argument("--workdir", default="/tmp/kisinwi_bench")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.workdir)
    out_root.mkdir(parents=True, exist_ok=True)

    folder = build_dataset_folder(args.key, out_root)
    zip_path = zip_folder(folder)
    if not args.no_upload:
        upload_to_platform(args.key, zip_path)
    # папку с распакованными изображениями чистим, zip оставляем
    shutil.rmtree(folder, ignore_errors=True)
    print(f"[{args.key}] готово")


if __name__ == "__main__":
    main()
