import sys
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Metric
from torchmetrics.classification import (
    Accuracy, Recall, Precision, F1Score
)
from torchmetrics import MeanMetric
from typing import Optional, Dict, List, Any, Union

from app.core.metrics import met_cl
from app.logs import get_logger

logger = get_logger(__name__)

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            
            # данные
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: Optional[DataLoader] = None,
            classes: Optional[List[str]] = None,
            
            # настройки тренировки
            loss_fn_config: Optional[Dict] = None,
            optimizer_config: Optional[Dict] = None,
            scheduler_config: Optional[Dict] = None,
            epochs: int = 10,
            device: str = 'cpu',
            test_every_n_epochs: int = 2,
            save_checkpoint_every_n_epochs: int = 5,
            checkpoint_dir: Optional[str] = None,
    ):
        logger.debug("⚪ Инициализация класса")

        # Модель и настройки обучения
        self.model = model
        self.loss_fn_config = loss_fn_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.epochs = epochs
        self.device = device
        self.test_every_n_epochs = test_every_n_epochs
        self.save_checkpoint_every_n_epochs = save_checkpoint_every_n_epochs
        self.checkpoint_dir = checkpoint_dir

        # Данные
        self.classes = classes if classes is not None else []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Валидация полученных данных
        self._validate_input()

        # Настройка устройства
        self._setup_device(device)
        self.model.to(self.device)

        # Настройка вспомогательных функция
        self._setup_loss_fn()
        self._setup_optimizer()
        self._setup_scheduler()

        # настройка метрик
        self.train_metrics = self._create_classification_metrics(
            preset='minimal',
            prefix='train_',
        )

        self.val_metrics = self._create_classification_metrics(
            preset='full',
            prefix='val_',
        )

        
        self.test_metrics = self._create_classification_metrics(
            preset='test',
            prefix='test_'
        )

        # История обучения
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

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
        ]

        for obj, expected_type, name in required_checks:
            if not isinstance(obj, expected_type):
                logger.error(f"└🔴 {name} is not {expected_type}. Got {type(obj)}")
                raise TypeError(f"{name} must be {expected_type}")

            if isinstance(obj, DataLoader) and len(obj) == 0:
                logger.error(f"└🔴 {name} is empty")
                raise ValueError(f"{name} cannot be empty")

            logger.debug(f"├🟢 {name}: OK")

        # Optional parameters
        if self.test_loader is not None and not isinstance(self.test_loader, DataLoader):
            logger.warning(f"🟠 test_loader is not DataLoader. Setting to None")
            self.test_loader = None

        if not isinstance(self.device, str):
            logger.warning(f"🟠 device is not str. Using 'cpu'")
            self.device = 'cpu'

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

    def _setup_device(self, device_str: str) -> None:
        """Setup device for training"""
        logger.debug("├🔘 Start setting device")

        if device_str == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("🟠 CUDA not available. Using CPU")
                self.device = torch.device('cpu')
            else:
                torch.cuda.empty_cache()
                gpu_name = torch.cuda.get_device_name(0)
                logger.debug(f"├🟡 GPU: {gpu_name}")
                self.device = torch.device('cuda')
        else:
            self.device = torch.device(device_str)

        logger.info(f"Training on: {self.device}")
        logger.debug(f"└🟢 Device set to: {self.device}")

    # __WARNING__ ВЫНЕСТИ В ОТДЕЛЬНУЮ ФУНКЦИЮ НАСТРОЙКУ МЕТРИК !!!!
    def _create_classification_metrics(
            self,
            preset: str = 'full',
            prefix: str = '',
    ) -> MetricCollection:
        """Create metrics collection based on preset"""
        num_classes = len(self.classes)
        
        # Disable sync for non-distributed training
        sync_on_compute = False

        metrics_dict: Dict[str, Union[Metric, MetricCollection]] = {}
        
        if preset == 'minimal':
            metrics_dict = {
                'accuracy': Accuracy(
                    task='multiclass',
                    num_classes=num_classes,
                    sync_on_compute=sync_on_compute
                ),
            }
        
        elif preset == 'standard':
            metrics_dict = {
                'accuracy': Accuracy(
                    task='multiclass',
                    num_classes=num_classes,
                    sync_on_compute=sync_on_compute
                ),
                'precision': Precision(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'recall': Recall(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'f1': F1Score(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
            }
        
        elif preset == 'full':
            metrics_dict = {
                'accuracy': Accuracy(
                    task='multiclass',
                    num_classes=num_classes,
                    sync_on_compute=sync_on_compute
                ),
                'precision_macro': Precision(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'precision_micro': Precision(
                    task='multiclass',
                    num_classes=num_classes,
                    average='micro',
                    sync_on_compute=sync_on_compute
                ),
                'recall_macro': Recall(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'recall_micro': Recall(
                    task='multiclass',
                    num_classes=num_classes,
                    average='micro',
                    sync_on_compute=sync_on_compute
                ),
                'f1_macro': F1Score(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'f1_micro': F1Score(
                    task='multiclass',
                    num_classes=num_classes,
                    average='micro',
                    sync_on_compute=sync_on_compute
                ),
            }
        
        elif preset == 'test':
            metrics_dict = {
                'accuracy': Accuracy(
                    task='multiclass',
                    num_classes=num_classes,
                    sync_on_compute=sync_on_compute
                ),
                'precision_macro': Precision(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'recall_macro': Recall(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
                'f1_macro': F1Score(
                    task='multiclass',
                    num_classes=num_classes,
                    average='macro',
                    sync_on_compute=sync_on_compute
                ),
            }
        
        else:
            raise ValueError(f"Unknown preset '{preset}'. Available: minimal, standard, full, test")
        
        # Add loss metric for non-test presets
        if preset != 'test':
            metrics_dict['loss'] = MeanMetric(
                sync_on_compute=sync_on_compute,
                nan_strategy='ignore'
            )
        
        # Create MetricCollection with proper typing
        collection = MetricCollection(
            metrics=metrics_dict,  # Explicitly name the parameter
            prefix=prefix,
        ).to(self.device)
        
        return collection

    def _train_one_epoch(self) -> Dict[str, Any]:
        """Train for one epoch"""
        logger.debug("🔘 Start epoch training")
        
        self.model.train()
        
        # Добавьте счетчик батчей
        batch_count = 0
        total_batches = len(self.train_loader)
        logger.debug(f"Total batches in train_loader: {total_batches}")
        
        try:
            for batch_idx, batch in enumerate(self._tqdm_loader(self.train_loader, "Training")):
                batch_count += 1
                
                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch
                else:
                    inputs, labels = batch, None
                
                inputs = inputs.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                if labels is not None:
                    loss = self.loss_fn(outputs, labels)
                else:
                    # For unsupervised or self-supervised learning
                    loss = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                if labels is not None:
                    _, predicted = torch.max(outputs, dim=1)
                    self.train_metrics.update(
                        preds=predicted,
                        target=labels,
                        value=loss.item()
                    )
                else:
                    # If no labels, just track loss
                    if hasattr(self.train_metrics, 'update'):
                        self.train_metrics.update(value=loss.item())
        
        except Exception as e:
            logger.error(f"Error during training at batch {batch_count}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        logger.debug(f"Processed {batch_count} batches out of {total_batches}")
        
        # Update scheduler
        if self.scheduler is not None:
            logger.debug("Updating scheduler")
            self.scheduler.step()
        
        # Compute metrics
        logger.debug("Computing training metrics")
        train_metrics_value = self.train_metrics.compute()
        
        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Loss train: {train_metrics_value.get('train_loss', 'N/A')}")
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Store in history
        if 'train_loss' in train_metrics_value:
            self.history['train_loss'].append(train_metrics_value['train_loss'])
        if 'train_accuracy' in train_metrics_value:
            self.history['train_accuracy'].append(train_metrics_value['train_accuracy'])
        self.history['learning_rate'].append(current_lr)
        
        # Reset metrics
        self.train_metrics.reset()
        
        # Clean CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug("🏁 Finish epoch training")
        return train_metrics_value

    def _validate_one_epoch(self) -> Dict[str, Any]:
        """Validate for one epoch"""
        logger.debug("🔘 Start validation")
        self.model.eval()
        
        with torch.no_grad():
            for batch in self._tqdm_loader(self.val_loader, "Validating"):
                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch
                else:
                    inputs, labels = batch, None
                
                inputs = inputs.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                
                # Calculate loss
                if labels is not None:
                    loss = self.loss_fn(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    self.val_metrics.update(
                        preds=predicted,
                        target=labels,
                        value=loss.item()
                    )
                else:
                    # If no labels, just track loss
                    loss = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                    if hasattr(self.val_metrics, 'update'):
                        self.val_metrics.update(value=loss.item())
        
        val_metrics_value = self.val_metrics.compute()
        logger.info(f"Validation loss: {val_metrics_value.get('val_loss', 'N/A')}")
        
        # Store in history
        if 'val_loss' in val_metrics_value:
            self.history['val_loss'].append(val_metrics_value['val_loss'])
        if 'val_accuracy' in val_metrics_value:
            self.history['val_accuracy'].append(val_metrics_value['val_accuracy'])
        
        self.val_metrics.reset()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug("🏁 Finish validation")
        return val_metrics_value

    def _test_model(self) -> Optional[Dict[str, Any]]:
        """Test the model on test dataset"""
        if self.test_loader is None or self.test_metrics is None:
            logger.warning("Test not performed: test_loader or test_metrics is None")
            return None
        
        logger.debug("⚪ Starting model testing")
        
        try:
            self.model.eval()
            self.test_metrics.reset()
            
            with torch.no_grad():
                for batch in self._tqdm_loader(self.test_loader, "Testing"):
                    # Unpack batch
                    if isinstance(batch, (list, tuple)):
                        inputs, labels = batch
                    else:
                        inputs, labels = batch, None
                    
                    inputs = inputs.to(self.device)
                    if labels is not None:
                        labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    
                    if labels is not None:
                        self.test_metrics.update(
                            preds=outputs,
                            target=labels
                        )
            
            test_metrics_value = self.test_metrics.compute()
            logger.info(f"Test results: {test_metrics_value}")
            logger.debug("🟢 Testing completed successfully")
            return test_metrics_value
            
        except Exception as e:
            logger.error(f"🔴 Testing failed: {e}")
            return None

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint"""
        if self.checkpoint_dir is None:
            return
        
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'history': self.history
            }
            
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"🔴 Failed to save checkpoint: {e}")

    def _log_epoch_metrics(
            self,
            epoch: int,
            train_metrics: Dict[str, Any],
            val_metrics: Dict[str, Any],
            test_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics for current epoch"""
        try:
            # Log to console
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch} Summary:")
            logger.info(f"  Train - Loss: {train_metrics.get('train_loss', 'N/A'):.4f}, "
                           f"Accuracy: {train_metrics.get('train_accuracy', 'N/A'):.4f}")
            logger.info(f"  Val   - Loss: {val_metrics.get('val_loss', 'N/A'):.4f}, "
                           f"Accuracy: {val_metrics.get('val_accuracy', 'N/A'):.4f}")
            
            if test_metrics:
                logger.info(f"  Test  - Accuracy: {test_metrics.get('test_accuracy', 'N/A'):.4f}, "
                               f"F1 Macro: {test_metrics.get('test_f1_macro', 'N/A'):.4f}")
            logger.info(f"{'='*50}\n")
            
            met_cl.log_epoch_metrics(
                "id_test", 
                epoch,
                train_metrics,
                val_metrics,
                test_metrics
            )
            
        except Exception as e:
            logger.error(f"🔴 Error logging metrics: {e}")

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
        logger.info("🔘 Старт тренировки")
        
        try:
            for epoch in range(1, self.epochs + 1):
                logger.info("=" * 40)
                logger.info(f"🔄 Эпоха [{epoch}/{self.epochs}]")
                
                try:
                    # Train and validate
                    logger.debug("Тренировочные данные...")
                    train_metrics = self._train_one_epoch()
                    logger.debug("Валидация...")
                    val_metrics = self._validate_one_epoch()
                except Exception as e:
                    logger.error(f"Error in epoch {epoch}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise
                
                
                test_metrics = None
                if (self.test_loader is not None and 
                        epoch % self.test_every_n_epochs == 0):
                    logger.debug("Старт теста...")
                    test_metrics = self._test_model()
                    logger.debug("Тестирование пройдено")
                
                # Log metrics
                self._log_epoch_metrics(epoch, train_metrics, val_metrics, test_metrics)
                
                # Save checkpoint
                if (self.checkpoint_dir is not None and 
                    epoch % self.save_checkpoint_every_n_epochs == 0):
                    self._save_checkpoint(epoch, {
                        'train': train_metrics,
                        'val': val_metrics,
                        'test': test_metrics
                    })
                
                logger.info(f"🟢 Epoch [{epoch}/{self.epochs}] completed")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        logger.info("🏁 Training finished successfully")
        return self.model
    