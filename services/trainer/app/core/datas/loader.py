import os
import torch
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import datasets

from app.api.schemas import DataLoaderParams
from .augmentations import build_transforms
from app.logs import get_logger

logger = get_logger(__name__)


def create_dataloaders(
    params: DataLoaderParams,
    base_data_dir: str = "datasets",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Создаём train/val/test DataLoader'ы для задачи классификации.

    Ожидаемая структура директорий:
        {base_data_dir}/{dataset_id}/{version_id}/
            train/
                class1/...
                class2/...
            val/
                class1/...
                class2/...
            test/
                class1/...
                class2/...

    Args:
        params: параметры загрузки данных
        base_data_dir: корневая папка с датасетами.

    Returns:
        Кортеж (train_loader, val_loader, test_loader, classes).
    """
    data_root = os.path.join(base_data_dir, params.dataset_id, params.version_id)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    # Проверяем существование обязательных папок
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(d):
            logger.error(f"Неверная структура датасета: {d} не найдена")
            raise FileNotFoundError(f"Неверная структура датасета: {d} не найдена")

    # Создание "трансформаторов"
    train_transform = build_transforms(params.train_transforms_config)
    val_test_transform = build_transforms(params.val_and_test_transforms_config)

    # Создание загрузчиков
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

    # Проверяем согласованность классов
    if not (train_dataset.classes == val_dataset.classes == test_dataset.classes):
        raise ValueError("Имена классов в каталогах не совпадают")
    classes = train_dataset.classes

    # pin_memory ускоряет передачу на GPU и бесполезен на CPU
    pin_memory = torch.cuda.is_available()
    # persistent_workers убирает пересоздание воркеров между эпохами
    persistent_workers = params.num_workers > 0

    # Создаём DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=params.num_workers,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=params.num_workers,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=params.num_workers,
        persistent_workers=persistent_workers
    )

    logger.info("✅ Датасеты собраны")
    logger.info(f"   Train выборка: {len(train_dataset)}")
    logger.info(f"   Val   выборка: {len(val_dataset)}")
    logger.info(f"   Test  выборка: {len(test_dataset)}")
    logger.info(f"   Классы:        {classes}")

    return train_loader, val_loader, test_loader, classes
