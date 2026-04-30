from typing import List
from torch import optim

def get_optimizers() -> List[str]:
    """
    Получение списка имеющихся оптимизаторов
    
    Список имеющихся классов оптимизаторов по ссылке
    https://github.com/pytorch/pytorch/blob/68e9155639a6e255ecab8f9b39be313297946c8c/torch/optim/__init__.py

    Оптимизаторы вычеркнутые из списка:
        * LBFGS - https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html
    """

    EXCLUDED_OPTIMIZERS = {
        'LBFGS'
    }

    return [name for name in optim.__all__ if name[0] == name[0].upper() and name not in EXCLUDED_OPTIMIZERS]

def get_schedulers() -> List[str]:
    """
    Получение списка планировщиков
    
    Список имеющихся классов планировщиков по ссылке
    https://github.com/pytorch/pytorch/blob/68e9155639a6e255ecab8f9b39be313297946c8c/torch/optim/lr_scheduler.py

    Планировщики вычеркнутые из списка:
        * ReduceLROnPlateau - https://docs.pytorch.org/docs/2.11/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    """

    EXCLUDED_SCHEDULERS = {
        'ReduceLROnPlateau'
    }

    return [name for name in optim.lr_scheduler.__all__ if name not in EXCLUDED_SCHEDULERS]
