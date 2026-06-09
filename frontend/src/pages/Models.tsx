import React, { useCallback, useEffect, useState } from 'react';
import { mlModelsService } from '../services/mlModelsService';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import { useModelFilters } from '../hooks';
import { MODELS_PAGE_SIZE } from '../constants';
import { ModelCard, ModelGroupCard } from '../components/models';
import Select from '../components/common/Select';
import type { MLModel, MLModelGroup, MLModelStatus } from '../types/mlModels';
import type { Dataset } from '../types/dataset';
import './Models.css';

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

  // Фильтры и пагинация.
  const { filters, offset, setFilter, setOffset, resetPage } = useModelFilters();
  const [statuses, setStatuses] = useState<MLModelStatus[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);

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
        name: filters.name || undefined,
        status: filters.status || undefined,
        dataset_id: filters.dataset || undefined,
        limit: MODELS_PAGE_SIZE,
        offset,
      });
      setModels(data.models);
      setFlatTotal(data.total);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модели', 'error');
    } finally {
      setLoading(false);
    }
  }, [filters, offset, showNotification]);

  const loadGrouped = useCallback(async () => {
    setLoading(true);
    try {
      const data = await mlModelsService.getGroupedModels({
        status: filters.status || undefined,
        dataset_id: filters.dataset || undefined,
        limit: MODELS_PAGE_SIZE,
        offset,
      });
      setGroups(data.groups);
      setGroupedTotal(data.total);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модели', 'error');
    } finally {
      setLoading(false);
    }
  }, [filters, offset, showNotification]);

  useEffect(() => {
    if (viewMode === 'flat') {
      loadFlat();
    } else {
      loadGrouped();
    }
  }, [viewMode, loadFlat, loadGrouped]);

  const switchView = (mode: ViewMode) => {
    resetPage();
    setViewMode(mode);
  };

  const total = viewMode === 'flat' ? flatTotal : groupedTotal;
  const page = Math.floor(offset / MODELS_PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(total / MODELS_PAGE_SIZE));
  const canPrev = offset > 0;
  const canNext = offset + MODELS_PAGE_SIZE < total;

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
                value={filters.name}
                onChange={(e) => setFilter('name', e.target.value)}
              />
            </div>
          )}

          <Select
            icon="fas fa-filter"
            ariaLabel="Фильтр по статусу"
            placeholder="Все статусы"
            value={filters.status}
            options={statuses.map((s) => ({ value: s.status, label: s.status }))}
            onChange={(v) => setFilter('status', v)}
          />

          <Select
            icon="fas fa-database"
            ariaLabel="Фильтр по датасету"
            placeholder="Все датасеты"
            value={filters.dataset}
            options={datasets.map((d) => ({ value: d.id, label: d.name }))}
            onChange={(v) => setFilter('dataset', v)}
          />
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
              onClick={() => setOffset(Math.max(0, offset - MODELS_PAGE_SIZE))}
            >
              <i className="fas fa-chevron-left"></i> Назад
            </button>
            <span className="pagination-info">
              Стр. {page} из {totalPages} · всего {total}
            </span>
            <button
              className="button secondary"
              disabled={!canNext}
              onClick={() => setOffset(offset + MODELS_PAGE_SIZE)}
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
