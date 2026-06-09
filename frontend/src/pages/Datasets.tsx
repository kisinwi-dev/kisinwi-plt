import React, { useState, useEffect } from 'react';
import { datasetService } from '../services/datasetService';
import type { Dataset, NewDataset, SourceItem } from '../types/dataset';
import './Datasets.css';
import { useNotification } from '../contexts/NotificationContext';
import { DatasetForm, DatasetCard, AddVersionForm } from '../components/datasets';

const EMPTY_SOURCE = (): SourceItem => ({ type: 'kaggle', url: null, description: '' });

// Идентификатор загрузки для нового датасета/версии.
const makeUploadId = () => `upload_${Date.now()}`;

const EMPTY_DATASET = () => ({
  name: '',
  description: '',
  type: 'image' as NewDataset['type'],
  task: 'classification' as NewDataset['task'],
  version: {
    name: '',
    description: '',
    sources: [EMPTY_SOURCE()],
  },
  file: null as File | null,
});

const EMPTY_VERSION = () => ({
  name: '',
  description: '',
  sources: [EMPTY_SOURCE()],
  file: null as File | null,
});

const Datasets: React.FC = () => {
  const { showNotification } = useNotification();

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showVersionForm, setShowVersionForm] = useState<string | null>(null);

  const [newDataset, setNewDataset] = useState(EMPTY_DATASET);
  const [newVersion, setNewVersion] = useState(EMPTY_VERSION);

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

  const handleVersionSourceAdd = () => {
    setNewDataset(prev => ({
      ...prev,
      version: { ...prev.version, sources: [...prev.version.sources, EMPTY_SOURCE()] },
    }));
  };

  const handleVersionSourceRemove = (index: number) => {
    if (newDataset.version.sources.length <= 1) return;
    setNewDataset(prev => ({
      ...prev,
      version: { ...prev.version, sources: prev.version.sources.filter((_, i) => i !== index) },
    }));
  };

  const handleVersionSourceChange = (index: number, field: keyof SourceItem, value: string) => {
    setNewDataset(prev => {
      const sources = [...prev.version.sources];
      sources[index] = { ...sources[index], [field]: field === 'url' ? (value || null) : value };
      return { ...prev, version: { ...prev.version, sources } };
    });
  };

  // ── Dataset CRUD ───────────────────────────────────────────────────────────

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
      setShowAddForm(false);
      showNotification('Датасет успешно создан', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка создания датасета', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDataset = async (id: string) => {
    if (!window.confirm('Вы уверены, что хотите удалить датасет?')) return;

    try {
      setLoading(true);
      const deleted = await datasetService.deleteDataset(id);
      if (!deleted) throw new Error('Не удалось удалить датасет');
      setDatasets(prev => prev.filter(ds => ds.id !== id));
      showNotification('Датасет удалён', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления датасета', 'error');
    } finally {
      setLoading(false);
    }
  };

  // ── Version CRUD ───────────────────────────────────────────────────────────

  const handleOpenVersionForm = (datasetId: string) => {
    setShowVersionForm(datasetId);
    setNewVersion(EMPTY_VERSION);
  };

  const handleAddVersion = async (datasetId: string) => {
    if (!newVersion.name) {
      showNotification('Введите название версии', 'warning');
      return;
    }

    try {
      setLoading(true);
      const id_data = makeUploadId();

      if (newVersion.file) {
        const uploaded = await datasetService.uploadFile(id_data, newVersion.file);
        if (!uploaded) throw new Error('Не удалось загрузить файл');
      }

      await datasetService.createVersion(datasetId, {
        id_data,
        name: newVersion.name,
        description: newVersion.description,
        sources: newVersion.sources,
      });

      const updated = await datasetService.getDatasets();
      setDatasets(updated);
      setShowVersionForm(null);
      showNotification('Версия успешно добавлена', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка добавления версии', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteVersion = async (datasetId: string, versionId: string) => {
    if (!window.confirm('Вы уверены, что хотите удалить эту версию?')) return;

    try {
      setLoading(true);
      const deleted = await datasetService.deleteVersion(datasetId, versionId);
      if (!deleted) throw new Error('Не удалось удалить версию');
      const updated = await datasetService.getDatasets();
      setDatasets(updated);
      showNotification('Версия удалена', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления версии', 'error');
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
        {!showAddForm && (
          <button className="button" onClick={() => setShowAddForm(true)} disabled={loading}>
            <i className="fas fa-plus"></i> Новый датасет
          </button>
        )}
      </div>

      {showAddForm && (
        <DatasetForm
          newDataset={newDataset}
          loading={loading}
          onNewDatasetChange={handleNewDatasetChange}
          onVersionChange={handleVersionFieldChange}
          onVersionSourceAdd={handleVersionSourceAdd}
          onVersionSourceRemove={handleVersionSourceRemove}
          onVersionSourceChange={handleVersionSourceChange}
          onFileSelect={(file) => setNewDataset(prev => ({ ...prev, file }))}
          onSubmit={handleCreateDataset}
          onCancel={() => setShowAddForm(false)}
        />
      )}

      <div className="datasets-list">
        {datasets.length === 0 && !loading ? (
          <p className="empty-state">Пока нет ни одного датасета. Создайте первый!</p>
        ) : (
          datasets.map(dataset => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              loading={loading}
              showVersionForm={showVersionForm === dataset.id}
              onAddVersion={() => handleOpenVersionForm(dataset.id)}
              onDelete={() => handleDeleteDataset(dataset.id)}
              onDeleteVersion={(versionId) => handleDeleteVersion(dataset.id, versionId)}
              versionForm={
                <AddVersionForm
                  version={newVersion}
                  loading={loading}
                  onVersionChange={setNewVersion}
                  onSubmit={() => handleAddVersion(dataset.id)}
                  onCancel={() => setShowVersionForm(null)}
                />
              }
            />
          ))
        )}
      </div>
    </div>
  );
};

export default Datasets;
