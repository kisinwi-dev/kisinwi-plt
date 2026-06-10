import React, { useCallback, useEffect, useState } from 'react';
import { mlModelsService } from '../services/mlModelsService';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import { useModelFilters, useDebouncedValue } from '../hooks';
import { MODELS_PAGE_SIZE } from '../constants';
import { ModelCard, ModelGroupCard } from '../components/models';
import Select from '../components/common/Select';
import { Tooltip } from '../components/common/Tooltip';
import type { MLModel, MLModelVersion, MLModelStatus } from '../types/mlModels';
import type { Dataset } from '../types/dataset';
import { ICONS } from '../constants/icons';
import './Models.css';

type ViewMode = 'grouped' | 'flat';

const Models: React.FC = () => {
  const { showNotification } = useNotification();

  const [viewMode, setViewMode] = useState<ViewMode>('grouped');

  // Flat mode state.
  const [versions, setVersions] = useState<MLModelVersion[]>([]);
  const [flatTotal, setFlatTotal] = useState(0);

  // Grouped mode state.
  const [models, setModels] = useState<MLModel[]>([]);
  const [groupedTotal, setGroupedTotal] = useState(0);

  const [loading, setLoading] = useState(false);

  // Фильтры и пагинация.
  const { filters, offset, setFilter, setOffset, resetPage } = useModelFilters();
  // Имя дебаунсим, чтобы не слать запрос на каждое нажатие клавиши.
  const debouncedName = useDebouncedValue(filters.name);
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
      const data = await mlModelsService.getVersions({
        name: debouncedName || undefined,
        status: filters.status || undefined,
        dataset_id: filters.dataset || undefined,
        limit: MODELS_PAGE_SIZE,
        offset,
      });
      setVersions(data.versions);
      setFlatTotal(data.total);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модели', 'error');
    } finally {
      setLoading(false);
    }
  }, [debouncedName, filters.status, filters.dataset, offset, showNotification]);

  const loadGrouped = useCallback(async () => {
    setLoading(true);
    try {
      const data = await mlModelsService.getModels({
        name: debouncedName || undefined,
        status: filters.status || undefined,
        dataset_id: filters.dataset || undefined,
        limit: MODELS_PAGE_SIZE,
        offset,
      });
      setModels(data.models);
      setGroupedTotal(data.total);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модели', 'error');
    } finally {
      setLoading(false);
    }
  }, [debouncedName, filters.status, filters.dataset, offset, showNotification]);

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

  const isEmpty = viewMode === 'flat' ? versions.length === 0 : models.length === 0;
  // Полноэкранный лоадер — только на первичной загрузке; при обновлении приглушаем текущий список.
  const initialLoading = loading && isEmpty;

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
          <div className="filter-field">
            <i className={`fas ${ICONS.search}`}></i>
            <input
              type="text"
              placeholder="Поиск по имени"
              value={filters.name}
              onChange={(e) => setFilter('name', e.target.value)}
            />
          </div>

          <Select
            icon={`fas ${ICONS.filter}`}
            ariaLabel="Фильтр по статусу"
            placeholder="Все статусы"
            value={filters.status}
            options={statuses.map((s) => ({ value: s.status, label: s.status }))}
            onChange={(v) => setFilter('status', v)}
          />

          <Select
            icon={`fas ${ICONS.dataset}`}
            ariaLabel="Фильтр по датасету"
            placeholder="Все датасеты"
            value={filters.dataset}
            options={datasets.map((d) => ({ value: d.id, label: d.name }))}
            onChange={(v) => setFilter('dataset', v)}
          />
        </div>

        <div className="view-toggle">
          <Tooltip content="Группировка по моделям">
            <button
              className={`view-toggle-btn${viewMode === 'grouped' ? ' active' : ''}`}
              onClick={() => switchView('grouped')}
              aria-label="Группировка по моделям"
            >
              <i className={`fas ${ICONS.groupedView}`}></i>
            </button>
          </Tooltip>
          <Tooltip content="Плоский список">
            <button
              className={`view-toggle-btn${viewMode === 'flat' ? ' active' : ''}`}
              onClick={() => switchView('flat')}
              aria-label="Плоский список"
            >
              <i className={`fas ${ICONS.listView}`}></i>
            </button>
          </Tooltip>
        </div>
      </div>

      {initialLoading ? (
        <div className="loading-state">
          <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка моделей…
        </div>
      ) : isEmpty ? (
        <div className="empty-state">
          <i className={`fas ${ICONS.empty}`}></i> Модели не найдены.
        </div>
      ) : (
        <>
          <div className={`models-grid${loading ? ' is-refreshing' : ''}`}>
            {viewMode === 'flat'
              ? versions.map((version) => <ModelCard key={version.id} model={version} />)
              : models.map((model) => <ModelGroupCard key={model.id} model={model} onReload={loadGrouped} />)
            }
          </div>

          <div className="models-pagination">
            <button
              className="button secondary"
              disabled={!canPrev}
              onClick={() => setOffset(Math.max(0, offset - MODELS_PAGE_SIZE))}
            >
              <i className={`fas ${ICONS.pagePrev}`}></i> Назад
            </button>
            <span className="pagination-info">
              Стр. {page} из {totalPages} · всего {total}
            </span>
            <button
              className="button secondary"
              disabled={!canNext}
              onClick={() => setOffset(offset + MODELS_PAGE_SIZE)}
            >
              Вперёд <i className={`fas ${ICONS.pageNext}`}></i>
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default Models;
