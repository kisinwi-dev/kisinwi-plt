import json
import os
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import threading
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError

from .schemas import MetricUpdate, MetricData, TaskMetrics, MetricsResponse


class MetricsStorage:
    """Класс для хранения и управления метриками моделей ML"""
    
    def __init__(self, storage_path: str = "metrics_storage"):
        """
        Инициализация хранилища метрик
        
        Args:
            storage_path: Путь к директории для хранения JSON файлов
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, TaskMetrics] = {}
        self._lock = threading.RLock()  # Для потокобезопасности
        
        # Загружаем существующие метрики при инициализации
        self._load_all_metrics()
    
    def _get_task_file_path(self, task_id: str) -> Path:
        """Получить путь к файлу метрик задачи"""
        return self.storage_path / f"{task_id}.json"
    
    def _load_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """Загрузить метрики задачи из файла"""
        file_path = self._get_task_file_path(task_id)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return TaskMetrics(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error loading metrics for task {task_id}: {e}")
            return None
    
    def _save_task_metrics(self, task_metrics: TaskMetrics) -> bool:
        """Сохранить метрики задачи в файл"""
        file_path = self._get_task_file_path(task_metrics.task_id)
        
        try:
            # Создаем резервную копию перед сохранением
            if file_path.exists():
                backup_path = file_path.with_suffix('.json.bak')
                file_path.rename(backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(
                    task_metrics.dict(), 
                    f, 
                    ensure_ascii=False, 
                    indent=2,
                    default=str  # Для обработки datetime и других типов
                )
            return True
        except Exception as e:
            print(f"Error saving metrics for task {task_metrics.task_id}: {e}")
            return False
    
    def _load_all_metrics(self):
        """Загрузить все существующие метрики из файлов в кэш"""
        with self._lock:
            for file_path in self.storage_path.glob("*.json"):
                if file_path.suffix == '.json' and not file_path.name.endswith('.bak'):
                    task_id = file_path.stem
                    task_metrics = self._load_task_metrics(task_id)
                    if task_metrics:
                        self._cache[task_id] = task_metrics
    
    def update_metric(self, metric_update: MetricUpdate) -> TaskMetrics:
        """
        Обновить или добавить метрику для задачи
        
        Args:
            metric_update: Данные метрики для обновления
            
        Returns:
            Обновленные метрики задачи
        """
        with self._lock:
            # Получаем существующие метрики или создаем новые
            if metric_update.task_id not in self._cache:
                task_metrics = self._load_task_metrics(metric_update.task_id)
                if task_metrics is None:
                    task_metrics = TaskMetrics(task_id=metric_update.task_id)
                self._cache[metric_update.task_id] = task_metrics
            
            task_metrics = self._cache[metric_update.task_id]
            
            # Получаем или создаем MetricData для данной метрики
            if metric_update.metric_name not in task_metrics.metrics:
                task_metrics.metrics[metric_update.metric_name] = MetricData()
            
            metric_data = task_metrics.metrics[metric_update.metric_name]
            
            # Добавляем новый шаг и значение
            metric_data.steps.append(metric_update.step)
            metric_data.values.append(metric_update.value)
            
            # Сохраняем в файл
            self._save_task_metrics(task_metrics)
            
            return task_metrics
    
    def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """
        Получить все метрики для задачи
        
        Args:
            task_id: Идентификатор задачи
            
        Returns:
            Метрики задачи или None, если задача не найдена
        """
        with self._lock:
            # Сначала проверяем кэш
            if task_id in self._cache:
                return self._cache[task_id]
            
            # Если нет в кэше, пробуем загрузить из файла
            task_metrics = self._load_task_metrics(task_id)
            if task_metrics:
                self._cache[task_id] = task_metrics
            
            return task_metrics
    
    def get_metric_data(self, task_id: str, metric_name: str) -> Optional[MetricData]:
        """
        Получить данные конкретной метрики для задачи
        
        Args:
            task_id: Идентификатор задачи
            metric_name: Имя метрики
            
        Returns:
            Данные метрики или None, если метрика не найдена
        """
        task_metrics = self.get_task_metrics(task_id)
        if task_metrics and metric_name in task_metrics.metrics:
            return task_metrics.metrics[metric_name]
        return None
    
    def get_all_tasks(self) -> List[str]:
        """
        Получить список всех задач
        
        Returns:
            Список ID всех задач
        """
        with self._lock:
            # Обновляем список из файлов
            existing_tasks = set()
            for file_path in self.storage_path.glob("*.json"):
                if file_path.suffix == '.json' and not file_path.name.endswith('.bak'):
                    existing_tasks.add(file_path.stem)
            
            # Добавляем задачи из кэша
            existing_tasks.update(self._cache.keys())
            
            return sorted(list(existing_tasks))
    
    def delete_task_metrics(self, task_id: str) -> bool:
        """
        Удалить все метрики для задачи
        
        Args:
            task_id: Идентификатор задачи
            
        Returns:
            True если успешно удалено, иначе False
        """
        with self._lock:
            # Удаляем из кэша
            if task_id in self._cache:
                del self._cache[task_id]
            
            # Удаляем файл
            file_path = self._get_task_file_path(task_id)
            if file_path.exists():
                try:
                    file_path.unlink()
                    return True
                except Exception as e:
                    print(f"Error deleting file for task {task_id}: {e}")
                    return False
            
            return True
    
    def delete_metric(self, task_id: str, metric_name: str) -> bool:
        """
        Удалить конкретную метрику из задачи
        
        Args:
            task_id: Идентификатор задачи
            metric_name: Имя метрики
            
        Returns:
            True если успешно удалено, иначе False
        """
        with self._lock:
            task_metrics = self.get_task_metrics(task_id)
            if task_metrics and metric_name in task_metrics.metrics:
                del task_metrics.metrics[metric_name]
                self._save_task_metrics(task_metrics)
                return True
            return False
    
    def clear_cache(self):
        """Очистить кэш (полезно для освобождения памяти)"""
        with self._lock:
            self._cache.clear()
    
    def backup_metrics(self, backup_path: str) -> bool:
        """
        Создать резервную копию всех метрик
        
        Args:
            backup_path: Путь для резервной копии
            
        Returns:
            True если успешно, иначе False
        """
        backup_dir = Path(backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for task_id in self.get_all_tasks():
                task_metrics = self.get_task_metrics(task_id)
                if task_metrics:
                    backup_file = backup_dir / f"{task_id}.json"
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(task_metrics.dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False