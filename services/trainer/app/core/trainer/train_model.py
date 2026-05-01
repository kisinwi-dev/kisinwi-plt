import sys
from tqdm import tqdm
import torch
from torch import nn, optim, device, Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any
import asyncio

from app.service.metrices import MetricesClient
from app.service.tasker import TaskerClient
from app.api.schemes import TrainerParams, LossConfig, OptimizerConfig, ShedulerConfig
from app.logs import get_logger


logger = get_logger(__name__)

class Trainer:
    def __init__(
            self,
            # Модель
            model: nn.Module,
            
            # Данные
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            classes: List[str],
            
            # Настройки тренировки
            train_params: TrainerParams,
            
            # Устройство
            device: device,

            # Вспомогательные сервисы
            tasker_service: TaskerClient,
            metric_service: MetricesClient,
    ):
        # Модель
        self.model = model
        # Устройство
        self.device = device

        # Данные
        self.classes = classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Валидация полученных данных
        self._validate_input()

        # Параметры обучения
        self.epochs = train_params.epochs
        self._setup_loss_fn(train_params.loss_fn)
        self._setup_optimizer(train_params.optimizer)
        self._setup_scheduler(train_params.scheduler)

        # Сервисы
        self._metric_service = metric_service
        self._tasker_service = tasker_service

        logger.debug("🏁 Инициализация успешно завершена")

    # __WARNING__ В БУДУЮЩЕМ НУЖНО ОБЯЗАТЕЛЬНО ВЫНЕСТИ ВСЮ ВАЛИДАЦИЮ ИЗ КЛАССА И СДЕЛАТЬ ОТДЕЛЬНЫМИ ФУНКЦИЯМИ ВАЛИДАЦИИ
    def _validate_input(self) -> None:
        """Валидация полученных параметров"""

        required_checks = [
            (self.model, nn.Module, "model"),
            (self.train_loader, DataLoader, "train_loader"),
            (self.val_loader, DataLoader, "val_loader"),
            (self.test_loader, DataLoader, "test_loader")
        ]

        for obj, expected_type, name in required_checks:
            if not isinstance(obj, expected_type):
                logger.error(f"{name} должен быть `{expected_type}`")
                raise TypeError(f"{name} должен быть `{expected_type}`")

            if isinstance(obj, DataLoader) and len(obj) == 0:
                logger.error(f"{name} не должен быть пустым")
                raise ValueError(f"{name} не должен быть пустым")

            logger.debug(f"✅ {name}: OK")

    def _setup_loss_fn(
            self,
            loss_config: LossConfig
        ) -> None:
        """Настройка функции потерь"""
        
        loss_fn_type = loss_config.name
        if not hasattr(nn, loss_fn_type):
            raise ValueError(f"Функция потерь '{loss_fn_type}' не найдена в torch.nn")
        
        loss_fn_params = loss_config.params
        
        # Конвертируем list в Tensor для весов классов
        if 'weight' in loss_fn_params:
            if isinstance(loss_fn_params['weight'], list):
                loss_fn_params['weight'] = torch.tensor(loss_fn_params['weight']).to(self.device)
                logger.debug(f"Конвертирован weight из list в Tensor: {loss_fn_params['weight'].shape}")
        
        # Для pos_weight в BCEWithLogitsLoss
        if 'pos_weight' in loss_fn_params:
            if isinstance(loss_fn_params['pos_weight'], list):
                loss_fn_params['pos_weight'] = torch.tensor(loss_fn_params['pos_weight']).to(self.device)
        
        loss_fn_class = getattr(nn, loss_fn_type)
        self.loss_fn = loss_fn_class(**loss_fn_params)
        
        logger.debug(f"Функция потерь `{loss_fn_type}` создана с параметрами: {loss_fn_params}")

    def _setup_optimizer(
            self,
            optimizer_config: OptimizerConfig
        ) -> None:
        """Настройка оптимизатора"""
        optimizer_type = optimizer_config.name
        
        if not hasattr(optim, optimizer_type):
            raise ValueError(f"Оптимизатор {optimizer_type} не найден в torch.optim")
        
        optimizer_params = optimizer_config.params
        optimizer_class = getattr(optim, optimizer_type)
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        
        logger.debug(f"Оптимизатор `{optimizer_type}` создан с параметрами: {optimizer_params}")

    def _setup_scheduler(
            self,
            scheduler_config: ShedulerConfig
        ) -> None:
        """Настройка планировщика"""
        
        scheduler_type = scheduler_config.name
        
        if not hasattr(lr_scheduler, scheduler_type):
            raise ValueError(f"Планировщик {scheduler_type} не найден в torch.optim.lr_scheduler")
        
        scheduler_params = scheduler_config.params
        
        scheduler_class = getattr(lr_scheduler, scheduler_type)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
        
        logger.debug(f"Планировщик `{scheduler_type}` создан с параметрами: {scheduler_params}")

    def _train_one_epoch(self):
        """Тренировка(одна эпоха)"""

        # __WARNING__ Требуется создать отдельную функцию для работы scheduler
        
        self.model.train()
        
        is_batch_scheduler = isinstance(
            self.scheduler,
            (lr_scheduler.OneCycleLR, lr_scheduler.CyclicLR)
        )

        try:
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

                if is_batch_scheduler:
                    self.scheduler.step()
                
                # Обновление метрик
                _, predicted = torch.max(outputs, dim=1)
                self._metric_service.update(
                    type='train',
                    preds=predicted,
                    targets=labels,
                    loss=loss
                )
            
            # Обновление scheduler
            if self.scheduler is not None and not is_batch_scheduler:
                logger.debug("Обновление scheduler")
                self.scheduler.step()

        except Exception as e:
            logger.error(f"Ошибка на стадии тренировки модели: {e}")
            raise
        
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
            try:
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
            except Exception as e:
                logger.error(f"Ошибка на этапе валидации модели: {e}")
                raise e
        
        # Расчёт метрик
        self._metric_service.compute('val')
        
        # Очистка кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _test_model(self) -> Optional[Dict[str, Any]]:
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
        """Получение обьекта tqdm"""
        return tqdm(
            data_loader,
            desc=desc,
            bar_format="{l_bar}{bar:20}{r_bar}",
            colour="blue",
            leave=False,
            file=sys.stdout
        )

    def _train_one_full_epoch(self):
        """Полное синхронное обучение"""
        logger.debug("Тренировка...")
        self._train_one_epoch()
        logger.debug("Валидация...")
        self._validate_one_epoch()

    async def train(
            self,
            value_procces_start: int,
            value_procces_end: int
        ) -> nn.Module:
        "Полный процесс тренировки"
        
        # Переменные для изсенения значений прогрессса
        value_procces = value_procces_start
        one_epoch_procces_value = (value_procces_end - value_procces_start) // self.epochs
        
        try:
            for epoch in range(1, self.epochs + 1):
                # Логгирование начала эпохи
                logger.info("=" * 40)
                text_info_start = f"🔄 Начало тренировки [{epoch}/{self.epochs}] эпохи"
                logger.info(text_info_start)
                
                # Обновление статуса задачи
                await self._tasker_service.update_status_task(value_procces + 1, description=text_info_start)
                
                # Тренировка и валидация
                await asyncio.to_thread(self._train_one_full_epoch)

                # Логгирование конца эпохи
                text_info_end = f"☑️ Эпоха [{epoch}/{self.epochs}] завершена"
                value_procces += one_epoch_procces_value
                logger.info(text_info_end)
                await self._tasker_service.update_status_task(value_procces - 1, description=text_info_end)

            logger.info("🦾 Тестирование модели...")
            await self._test_model()
            logger.info("✅ Тестирование завершено")

        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            raise
        
        return self.model