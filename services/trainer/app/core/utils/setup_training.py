from typing import List
from torch import optim

def get_optimizers() -> List[str]:
    """Получение списка имеющихся оптимизаторов"""
    # Список имеющихся классов оптимизации
    # https://github.com/pytorch/pytorch/blob/68e9155639a6e255ecab8f9b39be313297946c8c/torch/optim/__init__.py
    return [name for name in optim.__all__ if name[0] == name[0].upper()]

def get_schedulers() -> List[str]:
    """
    Получение списка планировщиков
    
    Планировщик ReduceLROnPlateau т.к. требует передачу в него метрик.
    """
    # Список имеющихся классов планировщиков
    # https://github.com/pytorch/pytorch/blob/68e9155639a6e255ecab8f9b39be313297946c8c/torch/optim/lr_scheduler.py

    EXCLUDED_SCHEDULERS = {
        'ReduceLROnPlateau',
        'OneCycleLR', 
        'CyclicLR'
    }

    return [name for name in optim.lr_scheduler.__all__ if name not in EXCLUDED_SCHEDULERS]
