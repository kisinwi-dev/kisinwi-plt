from typing import List, Tuple, Type

from torch import nn
from torch.utils.data import DataLoader

from app.logs import get_logger

logger = get_logger(__name__)


def validate_trainer_inputs(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
) -> None:
    """Валидация входных параметров Trainer"""

    required_checks: List[Tuple[object, Type, str]] = [
        (model, nn.Module, "model"),
        (train_loader, DataLoader, "train_loader"),
        (val_loader, DataLoader, "val_loader"),
        (test_loader, DataLoader, "test_loader")
    ]

    for obj, expected_type, name in required_checks:
        if not isinstance(obj, expected_type):
            logger.error(f"{name} должен быть `{expected_type}`")
            raise TypeError(f"{name} должен быть `{expected_type}`")

        if isinstance(obj, DataLoader) and len(obj) == 0:
            logger.error(f"{name} не должен быть пустым")
            raise ValueError(f"{name} не должен быть пустым")

        logger.debug(f"✅ {name}: OK")
