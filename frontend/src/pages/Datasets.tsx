import React, { useState, useEffect, useMemo } from 'react';
import { datasetService } from '../services/datasetService';
import type { Dataset, NewDataset, SourceItem } from '../types/dataset';
import './Datasets.css';
import { useNotification } from '../contexts/NotificationContext';
import { DatasetForm, DatasetCard } from '../components/datasets';
import Select from '../components/common/Select';
import ConfirmModal from '../components/common/ConfirmModal';

const TYPE_FILTER_OPTIONS = [
  { value: 'image', label: 'Image' },
  { value: 'text', label: 'Text' },
  { value: 'tabular', label: 'Tabular' },
  { value: 'other', label: 'Other' },
];

const TASK_FILTER_OPTIONS = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'detection', label: 'Detection' },
  { value: 'segmentation', label: 'Segmentation' },
  { value: 'other', label: 'Other' },
];

// Идентификатор загрузки для нового датасета.
const makeUploadId = () => `upload_${Date.now()}`;

const EMPTY_DATASET = () => ({
  name: '',
  description: '',
  type: 'image' as NewDataset['type'],
  task: 'classification' as NewDataset['task'],
  version: {
    name: '',
    description: '',
    sources: [] as SourceItem[],
  },
  file: null as File | null,
});

type DatasetsTab = 'create' | 'list';

const Datasets: React.FC = () => {
  const { showNotification } = useNotification();

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<DatasetsTab>('list');

  // Фильтры списка датасетов (клиентские — датасеты грузятся целиком).
  const [nameFilter, setNameFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [taskFilter, setTaskFilter] = useState('');

  const [newDataset, setNewDataset] = useState(EMPTY_DATASET);

  // Датасет, ожидающий подтверждения удаления (null — модалка закрыта).
  const [pendingDelete, setPendingDelete] = useState<Dataset | null>(null);

  const filteredDatasets = useMemo(() => {
    const name = nameFilter.trim().toLowerCase();
    return datasets.filter(ds =>
      (!name || ds.name.toLowerCase().includes(name)) &&
      (!typeFilter || ds.type === typeFilter) &&
      (!taskFilter || ds.task === taskFilter)
    );
  }, [datasets, nameFilter, typeFilter, taskFilter]);

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        setLoading(true);
        const data = await datasetService.getDatasets();
        setDatasets(data);
      } catch (err) {
        showNotification(err instanceof Error ? err.message : 'Ошибка загрузки датасетов', 'error');
      } finally {
        setLoading(false);
      }
    };
    fetchDatasets();
  }, [showNotification]);

  // ── Handlers: new dataset form ─────────────────────────────────────────────

  const handleNewDatasetChange = (field: keyof Omit<NewDataset, 'version'>, value: string) => {
    setNewDataset(prev => ({ ...prev, [field]: value }));
  };

  const handleVersionFieldChange = (field: 'name' | 'description' | 'sources', value: string) => {
    setNewDataset(prev => ({ ...prev, version: { ...prev.version, [field]: value } }));
  };

  const handleVersionSourcesChange = (sources: SourceItem[]) => {
    setNewDataset(prev => ({ ...prev, version: { ...prev.version, sources } }));
  };

  const handleCreateDataset = async () => {
    if (!newDataset.name || !newDataset.version.name) {
      showNotification('Заполните обязательные поля: название и название версии', 'warning');
      return;
    }

    try {
      setLoading(true);
      const id_data = makeUploadId();

      if (newDataset.file) {
        const uploaded = await datasetService.uploadFile(id_data, newDataset.file);
        if (!uploaded) throw new Error('Не удалось загрузить файл');
      }

      await datasetService.createDataset({
        name: newDataset.name,
        description: newDataset.description,
        type: newDataset.type,
        task: newDataset.task,
        version: {
          id_data,
          name: newDataset.version.name,
          description: newDataset.version.description,
          sources: newDataset.version.sources,
        },
      });

      const updated = await datasetService.getDatasets();
      setDatasets(updated);
      setNewDataset(EMPTY_DATASET);
      setActiveTab('list');
      showNotification('Датасет успешно создан', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка создания датасета', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete) return;
    const dataset = pendingDelete;
    setPendingDelete(null);
    try {
      setLoading(true);
      const deleted = await datasetService.deleteDataset(dataset.id);
      if (!deleted) throw new Error('Не удалось удалить датасет');
      setDatasets(prev => prev.filter(ds => ds.id !== dataset.id));
      showNotification('Датасет удалён', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления датасета', 'error');
    } finally {
      setLoading(false);
    }
  };

  // ── Main render ────────────────────────────────────────────────────────────

  if (loading && datasets.length === 0) {
    return <div className="loading-state">Загрузка датасетов...</div>;
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Управление датасетами</h1>
        <p className="page-description">
          Загружайте, удаляйте и управляйте версиями датасетов для классификации изображений.
        </p>
      </div>

      <div className="page-tabs">
        <button
          className={`page-tab ${activeTab === 'create' ? 'active' : ''}`}
          onClick={() => setActiveTab('create')}
        >
          <i className="fas fa-plus"></i> Создать датасет
        </button>
        <button
          className={`page-tab ${activeTab === 'list' ? 'active' : ''}`}
          onClick={() => setActiveTab('list')}
        >
          <i className="fas fa-list"></i> Список датасетов
        </button>
      </div>

      {activeTab === 'create' ? (
        <DatasetForm
          newDataset={newDataset}
          loading={loading}
          onNewDatasetChange={handleNewDatasetChange}
          onVersionChange={handleVersionFieldChange}
          onVersionSourcesChange={handleVersionSourcesChange}
          onFileSelect={(file) => setNewDataset(prev => ({ ...prev, file }))}
          onSubmit={handleCreateDataset}
          onCancel={() => setActiveTab('list')}
        />
      ) : (
        <>
          <div className="list-toolbar">
            <div className="list-filters">
              <div className="filter-field">
                <i className="fas fa-search"></i>
                <input
                  type="text"
                  placeholder="Поиск по имени"
                  value={nameFilter}
                  onChange={(e) => setNameFilter(e.target.value)}
                />
              </div>

              <Select
                icon="fas fa-shapes"
                ariaLabel="Фильтр по типу"
                placeholder="Все типы"
                value={typeFilter}
                options={TYPE_FILTER_OPTIONS}
                onChange={setTypeFilter}
              />

              <Select
                icon="fas fa-bullseye"
                ariaLabel="Фильтр по задаче"
                placeholder="Все задачи"
                value={taskFilter}
                options={TASK_FILTER_OPTIONS}
                onChange={setTaskFilter}
              />
            </div>
          </div>

          <div className="datasets-list">
            {datasets.length === 0 && !loading ? (
              <p className="empty-state">Пока нет ни одного датасета. Создайте первый!</p>
            ) : filteredDatasets.length === 0 ? (
              <p className="empty-state">Датасеты не найдены.</p>
            ) : (
              filteredDatasets.map(dataset => (
                <DatasetCard key={dataset.id} dataset={dataset} onDelete={setPendingDelete} />
              ))
            )}
          </div>
        </>
      )}

      <ConfirmModal
        open={pendingDelete !== null}
        danger
        title="Удалить датасет?"
        message={pendingDelete ? `Датасет «${pendingDelete.name}» будет удалён безвозвратно.` : undefined}
        confirmLabel="Удалить"
        cancelLabel="Отмена"
        onConfirm={handleConfirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
};

export default Datasets;
