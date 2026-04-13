from typing import Tuple, List, Optional
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from app.logs import get_logger

logger = get_logger(__name__)

# Константы ImageNet для нормализации по умолчанию
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def calculate_normalize_dataset(
    dataloader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычисляет среднее и стандартное отклонение по датасету.
    Используется для нормализации, если is_calculate_normalize_dataset=True.
    """
    logger.debug("⚪[calculate_normalize_dataset] start")
    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    num_batches = 0

    for data, _ in tqdm(dataloader, desc="Calculating normalization"):
        # data: (batch, channels, height, width)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sq_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Датасет пустой, нельзя рассчитать нормализацию")

    mean = channels_sum / num_batches
    std = (channels_sq_sum / num_batches - mean ** 2) ** 0.5
    logger.debug(f"🟢[calculate_normalize_dataset] computed mean={mean}, std={std}")
    return mean, std


def load_dataloaders(
    dataset_id: str,
    version_id: str,
    img_w_size: int,
    img_h_size: int,
    batch_size: int,
    is_calculate_normalize_dataset: bool = False,
    base_data_dir: str = "datasets",
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Загружает train/val/test DataLoader'ы для задачи классификации.

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
        dataset_id: идентификатор датасета.
        version_id: идентификатор версии.
        img_w_size: ширина изображения после ресайза.
        img_h_size: высота изображения после ресайза.
        batch_size: размер батча.
        is_calculate_normalize_dataset: если True, вычислить нормализацию на train датасете.
        base_data_dir: корневая папка с датасетами.

    Returns:
        Кортеж (train_loader, val_loader, test_loader, classes).
    """
    try:
        data_root = os.path.join(base_data_dir, dataset_id, version_id)
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
        test_dir = os.path.join(data_root, "test")

        # Проверяем существование обязательных папок
        for d in [train_dir, val_dir, test_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError("Датасет не найден.")

        # Базовые трансформации без нормализации (только ресайз и ToTensor)
        base_transform = transforms.Compose([
            transforms.Resize((img_h_size, img_w_size)),
            transforms.ToTensor(),
        ])

        # Временный датасет для вычисления нормализации (если нужно)
        if is_calculate_normalize_dataset:
            logger.info("Вычисление нормализации конкретного набора данных...")
            temp_train_dataset = datasets.ImageFolder(root=train_dir, transform=base_transform)
            temp_loader = DataLoader(temp_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            mean, std = calculate_normalize_dataset(temp_loader)
            logger.info(f"Нормализация имеет значения mean={mean.tolist()}, std={std.tolist()}")
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
            logger.info("Использовании нормализации из ImageNet")

        # Итоговые трансформации с нормализацией
        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((img_h_size, img_w_size), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((img_h_size, img_w_size)),
            transforms.ToTensor(),
            normalize,
        ])

        # Загружаем датасеты с финальными трансформациями
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

        # Проверяем согласованность классов
        if not (train_dataset.classes == val_dataset.classes == test_dataset.classes):
            raise ValueError("Имена классов в каталогах не совпадают")

        classes = train_dataset.classes

        # Создаём DataLoader'ы
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )

        logger.info("✅ Датасеты собраны")
        logger.info(f"   Train выборка: {len(train_dataset)}")
        logger.info(f"   Val   выборка: {len(val_dataset)}")
        logger.info(f"   Test  выборка: {len(test_dataset)}")
        logger.info(f"   Классы:        {classes}")

        return train_loader, val_loader, test_loader, classes
    except Exception as e:
        logger.error(f'Ошибка: {e}')
        raise e