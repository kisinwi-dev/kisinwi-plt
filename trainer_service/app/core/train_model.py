import sys
from tqdm import tqdm
import torch
from torch import nn, optim, device, Tensor
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any, Union

from app.service.metrices import MetricesClient
from app.service.tasker import TaskerClient
from app.logs import get_logger

logger = get_logger(__name__)

class Trainer:
    def __init__(
            self,
            # модель
            model: nn.Module,
            
            # данные
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            classes: Optional[List[str]],
            
            # настройки тренировки
            loss_fn_config: Optional[Dict],
            optimizer_config: Optional[Dict],
            scheduler_config: Optional[Dict],
            epochs: int,
            device: device,

            # Вспомогательные сервисы
            tasker_service: TaskerClient,
            metric_service: MetricesClient,
    ):
        logger.debug("⚪ Инициализация класса")

        # Модель
        self.model = model
        # Устройство
        self.device = device

        # Данные
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Параметры обучения
        self.loss_fn_config = loss_fn_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.epochs = epochs

        # Валидация полученных данных
        self._validate_input()

        # Настройка вспомогательных функция
        self._setup_loss_fn()
        self._setup_optimizer()
        self._setup_scheduler()

        # Сервисы
        self._metric_service = metric_service
        self._tasker_service = tasker_service

        logger.debug("🏁 Инициализация успешно завершена")

    # __WARNING__ В БУДУЮЩЕМ НУЖНО ОБЯЗАТЕЛЬНО ВЫНЕСТИ ВСЮ ВАЛИДАЦИЮ ИЗ КЛАССА И СДЕЛАТЬ ОТДЕЛЬНЫМИ ФУНКЦИЯМИ ВАЛИДАЦИИ
    def _validate_input(self) -> None:
        """Validate all input parameters"""
        logger.debug("├🔘 Start input value validation")

        # Required parameters with strict validation
        required_checks = [
            (self.model, nn.Module, "model"),
            (self.train_loader, DataLoader, "train_loader"),
            (self.val_loader, DataLoader, "val_loader"),
            (self.test_loader, DataLoader, "test_loader")
        ]

        for obj, expected_type, name in required_checks:
            if not isinstance(obj, expected_type):
                logger.error(f"└🔴 {name} is not {expected_type}. Got {type(obj)}")
                raise TypeError(f"{name} must be {expected_type}")

            if isinstance(obj, DataLoader) and len(obj) == 0:
                logger.error(f"└🔴 {name} is empty")
                raise ValueError(f"{name} cannot be empty")

            logger.debug(f"├🟢 {name}: OK")

        logger.debug("└🏁 finish validating params")

    def _setup_loss_fn(self) -> None:
        """Setup loss function from config or use default"""
        if self.loss_fn_config is None or not isinstance(self.loss_fn_config, Dict):
            logger.warning("🟠 loss_fn_config not provided. Using default CrossEntropyLoss")
            self.loss_fn = nn.CrossEntropyLoss()
            logger.debug("├🟢 loss_fn: CrossEntropyLoss (default)")
        else:
            self.loss_fn = self._create_loss_fn_from_config(self.loss_fn_config)
            logger.debug("├🟢 loss_fn: created from config")

    def _setup_optimizer(self) -> None:
        """Setup optimizer from config or use default"""
        if self.optimizer_config is None or not isinstance(self.optimizer_config, Dict):
            logger.warning("🟠 optimizer_config not provided. Using default Adam (lr=0.001)")
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            logger.debug("├🟢 optimizer: Adam (default)")
        else:
            self.optimizer = self._create_optimizer_from_config(self.optimizer_config)
            logger.debug("├🟢 optimizer: created from config")

    def _setup_scheduler(self) -> None:
        """Setup scheduler from config or use default"""
        if self.scheduler_config is None or not isinstance(self.scheduler_config, Dict):
            logger.warning("🟠 scheduler_config not provided. Using CosineAnnealingLR (T_max=50)")
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
            logger.debug("├🟢 scheduler: CosineAnnealingLR (default)")
        else:
            self.scheduler = self._create_scheduler_from_config(self.scheduler_config)
            logger.debug("├🟢 scheduler: created from config")

    def _create_loss_fn_from_config(self, loss_fn_config: Dict) -> nn.Module:
        """Create loss function from configuration dictionary"""
        loss_fn_type = loss_fn_config.get('type', 'CrossEntropyLoss')
        
        if not hasattr(nn, loss_fn_type):
            raise ValueError(f"Loss function {loss_fn_type} not found in torch.nn")
        
        if 'params' not in loss_fn_config:
            raise ValueError("Loss function config must contain 'params' key")
        
        loss_fn_params = loss_fn_config['params']
        
        if not isinstance(loss_fn_params, dict):
            raise ValueError("'params' must be a dictionary")
        
        loss_fn_class = getattr(nn, loss_fn_type)
        loss_fn = loss_fn_class(**loss_fn_params)
        
        logger.debug(f"├ Loss function {loss_fn_type} created with params: {loss_fn_params}")
        return loss_fn

    def _create_optimizer_from_config(self, optimizer_config: Dict) -> Optimizer:
        """Create optimizer from configuration dictionary"""
        optimizer_type = optimizer_config.get('type', 'Adam')
        
        if not hasattr(optim, optimizer_type):
            raise ValueError(f"Optimizer {optimizer_type} not found in torch.optim")
        
        if 'params' not in optimizer_config:
            raise ValueError("Optimizer config must contain 'params' key")
        
        optimizer_params = optimizer_config['params']
        
        if not isinstance(optimizer_params, dict):
            raise ValueError("'params' must be a dictionary")
        
        optimizer_class = getattr(optim, optimizer_type)
        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        
        logger.debug(f"├ Optimizer {optimizer_type} created with params: {optimizer_params}")
        return optimizer

    def _create_scheduler_from_config(self, scheduler_config: Dict) -> Optional[lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from configuration dictionary"""
        if scheduler_config is None:
            return None
        
        scheduler_type = scheduler_config.get('type')
        if not scheduler_type:
            raise ValueError("Scheduler config must contain 'type' key")
        
        if not hasattr(lr_scheduler, scheduler_type):
            raise ValueError(f"Scheduler {scheduler_type} not found in torch.optim.lr_scheduler")
        
        if 'params' not in scheduler_config:
            raise ValueError("Scheduler config must contain 'params' key")
        
        scheduler_params = scheduler_config['params']
        
        if not isinstance(scheduler_params, dict):
            raise ValueError("Scheduler 'params' must be a dictionary")
        
        scheduler_class = getattr(lr_scheduler, scheduler_type)
        scheduler = scheduler_class(self.optimizer, **scheduler_params)
        
        logger.debug(f"├ Scheduler {scheduler_type} created with params: {scheduler_params}")
        return scheduler    

    def _train_one_epoch(self):
        """Тренировка(одна эпоха)"""
        
        self.model.train()
        
        for batch in self._tqdm_loader(self.train_loader, "Тренировка"):
            
            inputs, labels = batch
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Расчёт loss
            labels = self._prepare_labels_for_loss(outputs, labels)
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Обновление метрик
            _, predicted = torch.max(outputs, dim=1)
            self._metric_service.update(
                type='train',
                preds=predicted,
                targets=labels,
                loss=loss
            )
        
        # Обновление scheduler
        if self.scheduler is not None:
            logger.debug("Обновление scheduler")
            self.scheduler.step()
        
        # Расчёт метрик
        self._metric_service.compute('train')

        # Очистка кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_labels_for_loss(
            self, 
            outputs: Tensor, 
            labels: Tensor
        ) -> Tensor:
        """
        Подготавливает labels в зависимости от типа loss функции
        
        Args:
            outputs: Выход модели [batch_size, num_classes] или [batch_size, 1]
            labels: Исходные метки (могут быть в разных форматах)
        
        Returns:
            Tensor: Подготовленные метки для loss функции
        """
        
        loss_type = self.loss_fn.__class__.__name__
        
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

    def _validate_one_epoch(self):
        """Валидация(одна эпоха)"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in self._tqdm_loader(self.val_loader, "Валидация"):
                inputs, labels = batch
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                
                # Рассчёт loss
                loss = self.loss_fn(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                # Обновление метрик
                self._metric_service.update(
                    type='val',
                    preds=predicted,
                    targets=labels,
                    loss=loss
                )
        
        # Расчёт метрик
        self._metric_service.compute('val')
        
        # Очистка кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _test_model(self) -> Optional[Dict[str, Any]]:
        """Тестирование модели на тестовых данных"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                for batch in self._tqdm_loader(self.test_loader, "Testing"):
                    
                    inputs, labels = batch
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    
                    self._metric_service.update(
                        'test',
                        preds=outputs,
                        targets=labels
                    )
            
            self._metric_service.compute('test')
            
        except Exception as e:
            logger.error(f"Ошибка на стадии тестирования: {e}")
            return None

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint"""
        pass
        # if self.checkpoint_dir is None:
        #     return
        
        # try:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': self.model.state_dict(),
        #         'optimizer_state_dict': self.optimizer.state_dict(),
        #         'metrics': metrics,
        #         'history': self.history
        #     }
            
        #     if self.scheduler is not None:
        #         checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        #     checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        #     torch.save(checkpoint, checkpoint_path)
            
        #     logger.info(f"Checkpoint saved: {checkpoint_path}")
        # except Exception as e:
        #     logger.error(f"🔴 Failed to save checkpoint: {e}")

    def _tqdm_loader(self, data_loader: DataLoader, desc: str = "process") -> tqdm:
        """Return tqdm-wrapped dataloader for progress display"""
        return tqdm(
            data_loader,
            desc=desc,
            bar_format="{l_bar}{bar:20}{r_bar}",
            colour="blue",
            leave=False,
            file=sys.stdout
        )

    def train(self) -> nn.Module:
        "Полный процесс тренировки"
        try:
            for epoch in range(1, self.epochs + 1):
                logger.info("=" * 40)
                logger.info(f"🔄 Эпоха [{epoch}/{self.epochs}]")
                
                # Train and validate
                logger.debug("Тренировка...")
                self._train_one_epoch()
                logger.debug("Валидация...")
                self._validate_one_epoch()
                
                logger.info(f"🟢 Эпоха [{epoch}/{self.epochs}] завершена")

            logger.info("🥳🥳 Тестирование модели... 🥳🥳")
            self._test_model()

        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            raise
        
        return self.model