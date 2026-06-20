import sys
import tempfile
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn, optim, device, Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import asyncio

from app.service.metrices import MetricesClient
from app.service.tasker import TaskerClient
from app.api.schemas import TrainerParams, LossConfig, OptimizerConfig, SchedulerConfig
from app.core.exceptions import TaskCancelledError
from app.logs import get_logger

from .label_preparation import prepare_labels_for_loss
from .validation import validate_trainer_inputs

logger = get_logger(__name__)

class Trainer:
    def __init__(
            self,
            # ID модели (для имени чекпоинта)
            model_id: str,

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
        validate_trainer_inputs(
            self.model, self.train_loader, self.val_loader, self.test_loader
        )

        # Параметры обучения
        self.epochs = train_params.epochs
        self._setup_loss_fn(train_params.loss_fn)
        self._setup_optimizer(train_params.optimizer)
        self._setup_scheduler(train_params.scheduler)

        # AMP и gradient clipping (AMP работает только на CUDA)
        self._use_amp = train_params.use_amp and self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler(self.device.type, enabled=self._use_amp)
        self._grad_clip_norm = train_params.grad_clip_norm
        if train_params.use_amp and not self._use_amp:
            logger.warning("AMP запрошен, но устройство не CUDA — обучение без AMP")

        # Направление улучшения метрики ранней остановки (для выбора лучшей эпохи)
        self._early_stop_mode = train_params.early_stop.mode

        # Сервисы
        self._metric_service = metric_service
        self._tasker_service = tasker_service

        # Чекпоинт лучшей эпохи
        self.checkpoint_path = Path(tempfile.gettempdir()) / "checkpoints" / f"model_{model_id}.pt"

        # Итог обучения: эпоха/значение лучшего чекпоинта (None — улучшение
        # не фиксировалось, сохранены веса финальной эпохи) и фактически
        # последняя отученная эпоха (early stop может прервать цикл раньше).
        self.best_epoch: Optional[int] = None
        self.best_value: Optional[float] = None
        self.last_epoch: int = 0

        logger.debug("🏁 Инициализация успешно завершена")

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
            scheduler_config: SchedulerConfig
        ) -> None:
        """Настройка планировщика"""

        scheduler_type = scheduler_config.name

        if not hasattr(lr_scheduler, scheduler_type):
            raise ValueError(f"Планировщик {scheduler_type} не найден в torch.optim.lr_scheduler")

        scheduler_params = scheduler_config.params

        scheduler_class = getattr(lr_scheduler, scheduler_type)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_params)

        logger.debug(f"Планировщик `{scheduler_type}` создан с параметрами: {scheduler_params}")

    def _is_batch_scheduler(self) -> bool:
        """Batch-планировщики шагают после каждого батча, остальные — после эпохи"""
        return isinstance(
            self.scheduler,
            (lr_scheduler.OneCycleLR, lr_scheduler.CyclicLR)
        )

    def _step_scheduler(self, after_epoch: bool) -> None:
        """Шаг планировщика: вызывается после батча (after_epoch=False) и после эпохи"""
        if after_epoch:
            if self.scheduler is not None and not self._is_batch_scheduler():
                logger.debug("Обновление scheduler")
                self.scheduler.step()
        elif self._is_batch_scheduler():
            self.scheduler.step()

    def _train_one_epoch(self):
        """Тренировка(одна эпоха)"""

        self.model.train()

        for batch in self._iter_with_progress(self.train_loader, "Тренировка"):

            inputs, labels = batch

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (с autocast при включённом AMP)
            with torch.amp.autocast(self.device.type, enabled=self._use_amp):
                outputs = self.model(inputs)

                # Расчёт loss (labels для loss готовятся отдельно,
                # в метрики уходят исходные индексы классов)
                loss_targets = prepare_labels_for_loss(self.loss_fn, outputs, labels)
                loss = self.loss_fn(outputs, loss_targets)

            # Backward pass
            self._scaler.scale(loss).backward()

            if self._grad_clip_norm is not None:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip_norm)

            self._scaler.step(self.optimizer)
            self._scaler.update()

            self._step_scheduler(after_epoch=False)

            # Обновление метрик: передаём логиты [N, C] — label-метрики берут
            # argmax сами, AUROC требует именно логиты/вероятности
            self._metric_service.update(
                type='train',
                preds=outputs.detach().float(),
                targets=labels,
                loss=loss
            )

        # Обновление scheduler
        self._step_scheduler(after_epoch=True)

        # Расчёт метрик
        self._metric_service.compute('train')

        # Очистка кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate_one_epoch(self) -> bool:
        """
        Валидация(одна эпоха)

        Returns:
            bool: True - стоит продолжать обучение. False - нет смысла продолжать обучение.
        """
        self.model.eval()

        with torch.no_grad():
            for batch in self._iter_with_progress(self.val_loader, "Валидация"):
                inputs, labels = batch

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)

                # Рассчёт loss
                loss_targets = prepare_labels_for_loss(self.loss_fn, outputs, labels)
                loss = self.loss_fn(outputs, loss_targets)

                # Обновление метрик (логиты [N, C], см. _train_one_epoch)
                self._metric_service.update(
                    type='val',
                    preds=outputs,
                    targets=labels,
                    loss=loss
                )

        # Расчёт метрик
        relevance = self._metric_service.compute('val')

        # Очистка кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return relevance

    def _test_model(self) -> None:
        """Тестирование модели на тестовых данных"""
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
        self._metric_service.send_class_report()

    def _iter_with_progress(self, data_loader: DataLoader, desc: str):
        """Итерация по батчам с логом прогресса каждые ~5% (tqdm не виден в контейнере)"""
        total = len(data_loader)
        step = max(1, total // 20)  # 5% = 1/20; при <20 батчах — каждый батч
        for i, batch in enumerate(self._tqdm_loader(data_loader, desc), start=1):
            yield batch
            if i % step == 0 or i == total:
                logger.info(f"{desc}: {round(i * 100 / total)}% ({i}/{total} батчей)")

    def _tqdm_loader(self, data_loader: DataLoader, desc: str = "process") -> tqdm:
        """Получение обьекта tqdm"""
        return tqdm(
            data_loader,
            desc=desc,
            bar_format="{l_bar}{bar:20}{r_bar}",
            colour="blue",
            leave=False,
            file=sys.stdout,
            disable=not sys.stdout.isatty()
        )

    def _train_one_full_epoch(self) -> bool:
        """
        Полное синхронное обучение

        Returns:
            bool: True - стоит продолжать обучение. False - нет смысла продолжать обучение.
        """
        logger.debug("Тренировка...")
        self._train_one_epoch()
        logger.debug("Валидация...")
        return self._validate_one_epoch()

    def _save_checkpoint(
            self,
            state_dict: Dict[str, Tensor],
            epoch: int,
            metric_value: float
    ) -> None:
        """Сохраняет веса лучшей эпохи на диск (ошибка сохранения не прерывает обучение)"""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": state_dict,
                    "epoch": epoch,
                    "metric": metric_value
                },
                self.checkpoint_path
            )
            logger.info(f"💾 Чекпоинт лучшей эпохи сохранён: {self.checkpoint_path}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить чекпоинт: {e!r}")

    def _is_better(self, current: float, best: Optional[float]) -> bool:
        """Сравнение значения метрики ранней остановки с лучшим"""
        if best is None:
            return True
        if self._early_stop_mode == 'max':
            return current > best
        return current < best

    async def train(
            self,
            progress_value_start: int,
            progress_value_end: int
        ) -> nn.Module:
        "Полный процесс тренировки"

        # Переменные для изменения значений прогресса
        progress_value = float(progress_value_start)
        progress_per_epoch = (progress_value_end - progress_value_start) / self.epochs

        # Лучшая эпоха по метрике ранней остановки (эпоха/значение — в self,
        # их после обучения отправляют в metrics; здесь только веса)
        best_state: Optional[Dict[str, Tensor]] = None

        for epoch in range(1, self.epochs + 1):
            self.last_epoch = epoch
            # Проверка отмены задачи (на границе эпох)
            if await self._tasker_service.get_task_status() == "cancelled":
                logger.info("🛑 Задача отменена пользователем, обучение остановлено")
                raise TaskCancelledError("Задача отменена пользователем")

            # Логгирование начала эпохи
            logger.info("=" * 40)
            text_info_start = f"🔄 [{epoch}/{self.epochs}] эпоха"
            logger.info(text_info_start)

            # Обновление статуса задачи
            await self._tasker_service.update_status_task(
                percentages=round(progress_value) + 1,
                status_info=text_info_start
            )

            # Тренировка и валидация
            relevance = await asyncio.to_thread(self._train_one_full_epoch)

            # Трекинг лучшей эпохи (веса храним на CPU, чтобы не занимать память GPU)
            current_value = self._metric_service.last_early_stop_value
            if current_value is not None and self._is_better(current_value, self.best_value):
                self.best_value = current_value
                self.best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                logger.info(f"🏅 Новая лучшая эпоха: {epoch} (метрика: {current_value:.6f})")
                await asyncio.to_thread(
                    self._save_checkpoint, best_state, self.best_epoch, self.best_value
                )

            # Логгирование конца эпохи
            text_info_end = f"☑️ Эпоха [{epoch}/{self.epochs}] завершена"
            progress_value += progress_per_epoch
            logger.info(text_info_end)
            await self._tasker_service.update_status_task(
                percentages=round(progress_value) - 1,
                status_info=text_info_end
            )

            if relevance == False:
                logger.info("Обучение модели остановлено")
                break

        # Восстановление весов лучшей эпохи
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"♻️ Восстановлены веса лучшей эпохи [{self.best_epoch}] (метрика: {self.best_value:.6f})")

        logger.info("🦾 Тестирование модели...")
        await asyncio.to_thread(self._test_model)
        logger.info("✅ Тестирование завершено")

        return self.model
