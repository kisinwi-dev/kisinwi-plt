import torch
from torch import device

from app.logs import get_logger

logger = get_logger(__name__)

def setup_device(device_str: str) -> device:
    """Найстройка и проверка вычислительных возможностей"""
    logger.debug("Проверка технических возможностей")

    if device_str == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("🟠 CUDA не найдена. Используем CPU")
            device = torch.device('cpu')
        else:
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            logger.debug(f"🟡 GPU: {gpu_name}")
            device = torch.device('cuda')
    else:
        device = torch.device(device_str)

    logger.info(f"Используется: {device}")
    return device