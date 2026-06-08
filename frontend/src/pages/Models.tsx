import React, { useCallback, useEffect, useState } from 'react';
import { mlModelsService } from '../services/mlModelsService';
import { useNotification } from '../contexts/NotificationContext';
import { ModelCard } from '../components/models';
import type { MLModel, MLModelStatus } from '../types/mlModels';
import './Models.css';

// Размер страницы списка моделей.
const PAGE_SIZE = 12;

const Models: React.FC = () => {
  const { showNotification } = useNotification();

  const [models, setModels] = useState<MLModel[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  // Фильтры.
  const [nameFilter, setNameFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [datasetFilter, setDatasetFilter] = useState('');
  const [statuses, setStatuses] = useState<MLModelStatus[]>([]);

  // Пагинация.
  const [offset, setOffset] = useState(0);

  // Загружаем справочник статусов один раз.
  useEffect(() => {
    mlModelsService.getModelStatuses()
      .then(setStatuses)
      .catch(() => { /* фильтр по статусу опционален — молча игнорируем */ });
  }, []);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await mlModelsService.getModels({
        name: nameFilter || undefined,
        status: statusFilter || undefined,
        dataset_id: datasetFilter || undefined,
        limit: PAGE_SIZE,
        offset,
      });
      setModels(data.models);
      setTotal(data.total);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модели', 'error');
    } finally {
      setLoading(false);
    }
  }, [nameFilter, statusFilter, datasetFilter, offset, showNotification]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  // Изменение любого фильтра сбрасывает страницу на первую.
  const resetAndSet = <T,>(setter: React.Dispatch<React.SetStateAction<T>>) => (value: T) => {
    setOffset(0);
    setter(value);
  };

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const canPrev = offset > 0;
  const canNext = offset + PAGE_SIZE < total;

  return (
    <div className="page">
      <div className="page-header">
        <h1>Модели</h1>
        <p className="page-description">
          Реестр обученных моделей: фильтруйте, открывайте детали и скачивайте файлы весов.
        </p>
      </div>

      <div className="models-filters">
        <div className="filter-field">
          <i className="fas fa-search"></i>
          <input
            type="text"
            placeholder="Поиск по имени"
            value={nameFilter}
            onChange={(e) => resetAndSet(setNameFilter)(e.target.value)}
          />
        </div>

        <div className="filter-field">
          <i className="fas fa-filter"></i>
          <select
            value={statusFilter}
            onChange={(e) => resetAndSet(setStatusFilter)(e.target.value)}
          >
            <option value="">Все статусы</option>
            {statuses.map((s) => (
              <option key={s.id} value={s.status}>{s.status}</option>
            ))}
          </select>
        </div>

        <div className="filter-field">
          <i className="fas fa-database"></i>
          <input
            type="text"
            placeholder="Dataset ID"
            value={datasetFilter}
            onChange={(e) => resetAndSet(setDatasetFilter)(e.target.value)}
          />
        </div>
      </div>

      {loading ? (
        <div className="loading-state">
          <i className="fas fa-spinner fa-spin"></i> Загрузка моделей…
        </div>
      ) : models.length === 0 ? (
        <div className="empty-state">
          <i className="fas fa-box-open"></i> Модели не найдены.
        </div>
      ) : (
        <>
          <div className="models-grid">
            {models.map((model) => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>

          <div className="models-pagination">
            <button
              className="button secondary"
              disabled={!canPrev}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              <i className="fas fa-chevron-left"></i> Назад
            </button>
            <span className="pagination-info">
              Стр. {page} из {totalPages} · всего {total}
            </span>
            <button
              className="button secondary"
              disabled={!canNext}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Вперёд <i className="fas fa-chevron-right"></i>
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default Models;
