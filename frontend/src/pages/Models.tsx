import React, { useCallback, useEffect, useState } from 'react';
import { mlModelsService } from '../services/mlModelsService';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import { ModelCard, ModelGroupCard } from '../components/models';
import type { MLModel, MLModelGroup, MLModelStatus } from '../types/mlModels';
import type { Dataset } from '../types/dataset';
import './Models.css';

const PAGE_SIZE = 12;

type ViewMode = 'grouped' | 'flat';

const Models: React.FC = () => {
  const { showNotification } = useNotification();

  const [viewMode, setViewMode] = useState<ViewMode>('grouped');

  // Flat mode state.
  const [models, setModels] = useState<MLModel[]>([]);
  const [flatTotal, setFlatTotal] = useState(0);

  // Grouped mode state.
  const [groups, setGroups] = useState<MLModelGroup[]>([]);
  const [groupedTotal, setGroupedTotal] = useState(0);

  const [loading, setLoading] = useState(false);

  // Фильтры.
  const [nameFilter, setNameFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [datasetFilter, setDatasetFilter] = useState('');
  const [statuses, setStatuses] = useState<MLModelStatus[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  // Пагинация.
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    mlModelsService.getModelStatuses()
      .then(setStatuses)
      .catch(() => { /* фильтр по статусу опционален */ });
    datasetService.getDatasets()
      .then(setDatasets)
      .catch(() => { /* фильтр по датасету опционален */ });
  }, []);

  const loadFlat = useCallback(async () => {
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
      setFlatTotal(data.total);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модели', 'error');
    } finally {
      setLoading(false);
    }
  }, [nameFilter, statusFilter, datasetFilter, offset, showNotification]);

  const loadGrouped = useCallback(async () => {
    setLoading(true);
    try {
      const data = await mlModelsService.getGroupedModels({
        status: statusFilter || undefined,
        dataset_id: datasetFilter || undefined,
        limit: PAGE_SIZE,
        offset,
      });
      setGroups(data.groups);
      setGroupedTotal(data.total);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модели', 'error');
    } finally {
      setLoading(false);
    }
  }, [statusFilter, datasetFilter, offset, showNotification]);

  useEffect(() => {
    if (viewMode === 'flat') {
      loadFlat();
    } else {
      loadGrouped();
    }
  }, [viewMode, loadFlat, loadGrouped]);

  const resetAndSet = <T,>(setter: React.Dispatch<React.SetStateAction<T>>) => (value: T) => {
    setOffset(0);
    setter(value);
  };

  const switchView = (mode: ViewMode) => {
    setOffset(0);
    setViewMode(mode);
  };

  const total = viewMode === 'flat' ? flatTotal : groupedTotal;
  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const canPrev = offset > 0;
  const canNext = offset + PAGE_SIZE < total;

  const isEmpty = viewMode === 'flat' ? models.length === 0 : groups.length === 0;

  return (
    <div className="page">
      <div className="page-header">
        <h1>Модели</h1>
        <p className="page-description">
          Реестр обученных моделей: фильтруйте, открывайте детали и скачивайте файлы весов.
        </p>
      </div>

      <div className="models-toolbar">
        <div className="models-filters">
          {viewMode === 'flat' && (
            <div className="filter-field">
              <i className="fas fa-search"></i>
              <input
                type="text"
                placeholder="Поиск по имени"
                value={nameFilter}
                onChange={(e) => resetAndSet(setNameFilter)(e.target.value)}
              />
            </div>
          )}

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
            <select
              value={datasetFilter}
              onChange={(e) => resetAndSet(setDatasetFilter)(e.target.value)}
            >
              <option value="">Все датасеты</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="view-toggle">
          <button
            className={`view-toggle-btn${viewMode === 'grouped' ? ' active' : ''}`}
            onClick={() => switchView('grouped')}
            title="Группировка по моделям"
          >
            <i className="fas fa-layer-group"></i>
          </button>
          <button
            className={`view-toggle-btn${viewMode === 'flat' ? ' active' : ''}`}
            onClick={() => switchView('flat')}
            title="Плоский список"
          >
            <i className="fas fa-list"></i>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="loading-state">
          <i className="fas fa-spinner fa-spin"></i> Загрузка моделей…
        </div>
      ) : isEmpty ? (
        <div className="empty-state">
          <i className="fas fa-box-open"></i> Модели не найдены.
        </div>
      ) : (
        <>
          <div className="models-grid">
            {viewMode === 'flat'
              ? models.map((model) => <ModelCard key={model.id} model={model} />)
              : groups.map((group) => <ModelGroupCard key={group.name} group={group} onReload={loadGrouped} />)
            }
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
