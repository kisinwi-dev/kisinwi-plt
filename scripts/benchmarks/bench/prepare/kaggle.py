"""Собрать 6 скачанных Kaggle-датасетов в единый набор формата платформы.

Повторяет раскладку референс-ноутбука (deepfake-image-detection.ipynb): пул из 6
датасетов, стратифицированный split 80/10/10 (seed=42), классы Real(0)/Fake(1).
Метки берутся по структуре папок: в DS1/DS4 колонка label в CSV инвертирована.
wild-deepfake/test не используется - у платформы только train/val/test.

Выход: <out>/{train,val,test}/{Real,Fake}/*.jpg (hardlink, без дублей), затем zip
и upload в datasets. Датасет регистрируется в data/datasets_registry.json под
ключом 'master_kaggle'.

Точка входа - prepare_kaggle(); управление через `benchmark.py master_kaggle`
(сырьё качается автоматически). Self-check: python -c "from bench.prepare.kaggle import demo; demo()"
"""
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from bench import load_env, DATA_DIR

load_env()  # .env до импорта config: иначе override BENCH_*_URL не подхватится

from bench.config import DATASETS, FRONTEND_URL, DATASETS_URL
from bench.common import (
    upload_archive, create_dataset, find_dataset_by_name,
    load_registry, save_registry, zip_folder,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEED = 42

KEY = "master_kaggle"
CFG = DATASETS[KEY]
LABEL_NAME = {0: "Real", 1: "Fake"}  # метки как в ноутбуке


def _sources(data_dir: Path):
    """(source_tag, label, dir) для 6 датасетов. Метка по папке: 0=Real, 1=Fake.

    CSV из DS1/DS4 игнорируем: разметка по папкам real/fake корректнее.
    """
    d = data_dir
    items = []

    # DS1 - 140k Real and Fake Faces
    base = d / "140k-real-and-fake-faces/real_vs_fake/real-vs-fake"
    for split in ("train", "valid", "test"):
        items.append(("140k", 0, base / split / "real"))
        items.append(("140k", 1, base / split / "fake"))

    # DS2 - manjilkarki deepfake-and-real-images
    base = d / "deepfake-and-real-images/Dataset"
    for split in ("Train", "Validation", "Test"):
        items.append(("manjil", 0, base / split / "Real"))
        items.append(("manjil", 1, base / split / "Fake"))

    # DS3 - ciplab (берём real_and_fake_face/, дубликат real_and_fake_face_detection/ игнорируем)
    base = d / "real-and-fake-face-detection/real_and_fake_face"
    items.append(("ciplab", 0, base / "training_real"))
    items.append(("ciplab", 1, base / "training_fake"))

    # DS4 - rvf10k
    base = d / "rvf10k/rvf10k"
    for split in ("train", "valid"):
        items.append(("rvf10k", 0, base / split / "real"))
        items.append(("rvf10k", 1, base / split / "fake"))

    # DS5 - deepfake-vs-real-20k
    base = d / "deepfake-vs-real-20k/Deep-vs-Real"
    items.append(("deepfake20k", 0, base / "Real"))
    items.append(("deepfake20k", 1, base / "Deepfake"))

    # DS6 - wild-deepfake (train+valid; test не берём)
    base = d / "wild-deepfake"
    for split in ("train", "valid"):
        items.append(("wild", 0, base / split / "real"))
        items.append(("wild", 1, base / split / "fake"))

    return items


def collect_records(data_dir: Path):
    """Собрать [{filepath, label, source}] по всем источникам, dedup по пути."""
    seen = set()
    records = []
    for src, label, folder in _sources(data_dir):
        if not folder.exists():
            print(f"  пропуск (нет папки): {folder}")
            continue
        for fp in folder.rglob("*"):
            if fp.suffix.lower() not in IMG_EXTS:
                continue
            key = str(fp.resolve())
            if key in seen:
                continue
            seen.add(key)
            records.append({"filepath": str(fp), "label": label, "source": src})
    return records


def stratified_split(records, fractions=(0.8, 0.1, 0.1), seed=SEED):
    """Делит records на (train, val, test) стратифицированно по label.

    Доли применяются внутри каждого класса, распределение сохраняется.
    """
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    train, val, test = [], [], []
    rng = random.Random(seed)
    for label, recs in by_class.items():
        recs = list(recs)
        rng.shuffle(recs)
        n = len(recs)
        n_train = int(n * fractions[0])
        n_val = int(n * fractions[1])
        train += recs[:n_train]
        val += recs[n_train:n_train + n_val]
        test += recs[n_train + n_val:]
    return train, val, test


def _place(record, dst: Path):
    """Hardlink (без дублирования места), на другой ФС - copy2."""
    src = record["filepath"]
    try:
        os.link(src, dst)
    except FileExistsError:
        pass
    except OSError:
        shutil.copy2(src, dst)


def build_folder(splits: dict, out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split, recs in splits.items():
        # счётчик на (split, source, label): basename'ы датасетов совпадают,
        # иначе коллизии имён теряют файлы
        counters = Counter()
        for r in recs:
            cls = LABEL_NAME[r["label"]]
            dst_dir = out_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            n = counters[(r["source"], r["label"])]
            counters[(r["source"], r["label"])] += 1
            ext = Path(r["filepath"]).suffix.lower()
            _place(r, dst_dir / f"{split}_{r['source']}_{r['label']}_{n}{ext}")
        print(f"  {split:<6}: {len(recs):,} файлов разложено")
    return out_dir


def upload_to_platform(zip_path: Path):
    id_data = "grandmaster_deepfake"
    platform_name = CFG["platform_name"]

    reg = load_registry()
    if KEY in reg and find_dataset_by_name(platform_name):
        print(f"[{KEY}] уже зарегистрирован ({reg[KEY]['dataset_id']}) - пропуск")
        return reg[KEY]

    print(f"[upload] {zip_path.name} ({zip_path.stat().st_size / 1e9:.1f} GB) "
          f"-> {DATASETS_URL}/upload ...")
    upload_archive(id_data, str(zip_path))

    payload = {
        "name": platform_name,
        "description": CFG["role"],
        "type": "image",
        "task": "classification",
        "version": {
            "id_data": id_data,
            "name": "v1",
            "description": "pooled 6 datasets, stratified 80/10/10, labels by folder",
            "sources": [
                {"type": "kaggle", "url": f"https://www.kaggle.com/datasets/{ref}"}
                for ref in CFG["kaggle_refs"]
            ],
        },
    }
    print("[upload] создание датасета ...")
    create_dataset(payload)

    info = find_dataset_by_name(platform_name)
    if info is None:
        raise RuntimeError(f"датасет '{platform_name}' создан, но не найден в списке")
    print(f"[upload] dataset_id={info['id']} классов={info.get('classes_count')}")
    print(f"[upload] {FRONTEND_URL}/datasets/{info['id']}")

    reg[KEY] = {
        "dataset_id": info["id"],
        "version_id": info["default_version_id"],
        "name": platform_name,
        "num_classes": info.get("classes_count", CFG["num_classes"]),
    }
    save_registry(reg)
    return reg[KEY]


def print_summary(records, splits):
    n_real = sum(1 for r in records if r["label"] == 0)
    n_fake = len(records) - n_real
    print(f"\nВсего: {len(records):,}  Real={n_real:,}  Fake={n_fake:,}")
    print("Per-source (Real / Fake):")
    by = defaultdict(lambda: [0, 0])
    for r in records:
        by[r["source"]][r["label"]] += 1
    for src in sorted(by):
        print(f"  {src:<12}: {by[src][0]:>8,} / {by[src][1]:>8,}")
    print("Splits (Real / Fake):")
    for name, recs in splits.items():
        rr = sum(1 for r in recs if r["label"] == 0)
        print(f"  {name:<6}: {len(recs):>8,}  ({rr:,} / {len(recs) - rr:,})")


def demo():
    """Self-check логики split: доли ~80/10/10, стратификация, без пересечений."""
    recs = [{"filepath": f"/x/{i}.jpg", "label": i % 2, "source": "s"}
            for i in range(1000)]
    train, val, test = stratified_split(recs)
    assert len(train) + len(val) + len(test) == 1000
    assert abs(len(train) - 800) <= 1 and abs(len(val) - 100) <= 1, (len(train), len(val))
    # стратификация: 50/50 в каждом сплите
    for s in (train, val, test):
        reals = sum(1 for r in s if r["label"] == 0)
        assert abs(reals - len(s) / 2) <= 1, (reals, len(s))
    # без пересечений
    paths = [r["filepath"] for s in (train, val, test) for r in s]
    assert len(paths) == len(set(paths))
    # детерминизм по seed
    assert [r["filepath"] for r in stratified_split(recs)[0]] == \
           [r["filepath"] for r in train]
    print("self-check OK")


def prepare_kaggle(data_dir: Path = DATA_DIR / "datasets",
                   out_dir: Path = DATA_DIR / "unified",
                   upload: bool = True, reuse_zip: bool = False) -> dict | None:
    """Собрать единый набор из Kaggle-датасетов и (опц.) загрузить.

    reuse_zip - не пересобирать уже готовый zip (~десятки ГБ) ради повторного upload.
    Возвращает запись реестра при upload, иначе None.
    """
    zip_path = out_dir.with_suffix(".zip")

    if reuse_zip:
        if not zip_path.exists():
            raise SystemExit(f"{zip_path} нет - убери reuse_zip, чтобы собрать заново")
        print(f"reuse zip {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"Сбор записей из {data_dir}/ ...")
        records = collect_records(data_dir)
        if not records:
            # Сырьё не найдено - качаем автоматически (download_kaggle идемпотентна).
            print("Сырьё Kaggle не найдено - загрузка (нужен KAGGLE_API_TOKEN) ...")
            from bench.prepare.download import download_kaggle
            download_kaggle(data_dir, unzip=True)
            records = collect_records(data_dir)
        if not records:
            raise SystemExit("0 записей даже после загрузки - проверь Kaggle-доступ и data_dir")
        train, val, test = stratified_split(records)
        splits = {"train": train, "val": val, "test": test}
        print_summary(records, splits)
        print(f"\nРаскладка в {out_dir}/ ...")
        build_folder(splits, out_dir)
        zip_path = zip_folder(out_dir)

    if not upload:
        print("upload пропущен")
        return None
    entry = upload_to_platform(zip_path)
    print("Готово.")
    return entry
