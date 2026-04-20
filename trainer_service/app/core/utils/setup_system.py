import torch
import platform
import psutil
from cpuinfo import get_cpu_info as get_for_cpu_info
from torch import device

from app.logs import get_logger

logger = get_logger(__name__)

def setup_device(device_str: str) -> device:
    """
    Настройка и проверка вычислительных возможностей
    """
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

def get_system_info() -> dict:
    """
    Получение полной информации о возможностях устройства.
    
    Проверка:
        + CPU
        + CUDA
        - XPU
        - mps
        - vulkan
    
    """
    
    info = {
        'platform': platform.platform(),
        'devices': []
    }

    info['devices'].append(get_cpu_info())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.debug(f"Сбор информации о 'GPU:{i}'...")
            props = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'type': 'cuda',
                'name': props.name,
                'memory_gb': round(props.total_memory / 1e9, 2),
                'index': i
            })
            logger.debug(f"✅ Информания по 'GPU:{i}' собрана")

    return info

def get_cpu_info() -> dict:
    """Получение информации о CPU"""
    logger.debug("Сбор информации о 'CPU'...")
    info = {}

    try:
        cpu_info_all = get_for_cpu_info()
        info['type'] = 'cpu',
        info['name'] = cpu_info_all['brand_raw']
        info['arch'] = cpu_info_all['arch']
        info['count'] = cpu_info_all['count']
        info['ram'] = round(psutil.virtual_memory().total / (1024**3), 2)
        logger.debug("✅ Информания по 'CPU' собрана")
    except Exception as e:
        logger.error(f"Не удалось получить информацию о CPU: {e}")
        info = {"type": "CPU not found"}

    return info
