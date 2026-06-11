import torch
from torch import nn, Tensor

from app.logs import get_logger

logger = get_logger(__name__)


def prepare_labels_for_loss(
        loss_fn: nn.Module,
        outputs: Tensor,
        labels: Tensor
) -> Tensor:
    """
    Подготавливает labels в зависимости от типа loss функции

    Args:
        loss_fn: Функция потерь (по её классу выбирается формат меток)
        outputs: Выход модели [batch_size, num_classes] или [batch_size, 1]
        labels: Исходные метки (могут быть в разных форматах)

    Returns:
        Tensor: Подготовленные метки для loss функции
    """

    loss_type = loss_fn.__class__.__name__

    # Для CrossEntropyLoss - нужны индексы классов
    if loss_type in ['CrossEntropyLoss', 'CrossEntropyLoss2d']:
        if labels.dim() == 2 and labels.size(1) > 1:
            labels = labels.argmax(dim=1)
        return labels.long()

    # Для BCEWithLogitsLoss и BCELoss
    elif loss_type in ['BCEWithLogitsLoss', 'BCELoss']:
        if outputs.size(1) == 1:
            if labels.dim() == 1:
                return labels.float().unsqueeze(1)
            elif labels.dim() == 2 and labels.size(1) > 1:
                return labels.argmax(dim=1).float().unsqueeze(1)
            else:
                return labels.float()

        # Бинарная классификация с двумя выходами [batch, 2]
        elif outputs.size(1) == 2:
            if labels.dim() == 1:
                labels_one_hot = torch.zeros(
                    labels.size(0), outputs.size(1),
                    device=labels.device
                )
                labels_one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
                return labels_one_hot.float()
            elif labels.dim() == 2 and labels.size(1) == 2:
                return labels.float()
            elif labels.dim() == 2 and labels.size(1) == 1:
                labels_one_hot = torch.zeros(
                    labels.size(0), outputs.size(1),
                    device=labels.device
                )
                labels_one_hot.scatter_(1, labels.long(), 1)
                return labels_one_hot.float()

        # Мульти классификация [batch, num_classes]
        else:
            if labels.dim() == 1:
                labels_one_hot = torch.zeros(
                    labels.size(0), outputs.size(1),
                    device=labels.device
                )
                labels_one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)
                return labels_one_hot.float()
            elif labels.dim() == 2 and labels.size(1) == outputs.size(1):
                return labels.float()

    # Для MSE Loss (регрессия)
    elif loss_type in ['MSELoss', 'L1Loss']:
        if labels.dim() != outputs.dim():
            if labels.dim() == 1 and outputs.dim() == 2:
                return labels.float().unsqueeze(1)
        return labels.float()

    # Не обработанная функция потерь
    else:
        logger.warning(f"Неизвестная loss функция: {loss_type}")
        return labels

    return labels
