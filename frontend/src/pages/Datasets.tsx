import React, { useState, useRef, useEffect } from 'react';
import { datasetService } from '../services/datasetService';
import type { Dataset, NewDataset, NewVersion, SourceItem } from '../types/dataset';
import DatasetForm from '../components/datasets/DatasetForm';
import DatasetCard from '../components/datasets/DatasetCard';
import AddVersionForm from '../components/datasets/AddVersionForm';
import './Datasets.css';

const Datasets: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showVersionForm, setShowVersionForm] = useState<string | null>(null);

  // Состояние для новой формы датасета
  const [newDataset, setNewDataset] = useState<NewDataset & { file: File | null }>({
    dataset_id: '',
    name: '',
    description: '',
    class_names: [],
    sources: [{ type: 'kaggle', url: '', description: '' }],
    type: 'image',
    task: 'classification',
    version: { version_id: '', description: '' },
    file: null,
  });

  // Состояние для ввода нового класса
  const [newClassName, setNewClassName] = useState('');
  const classInputRef = useRef<HTMLInputElement>(null);

  // Состояние для формы новой версии
  const [newVersion, setNewVersion] = useState<NewVersion & { file: File | null }>({
    version_id: '',
    description: '',
    file: null,
  });

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        setLoading(true);
        const data = await datasetService.getDatasets();
        setDatasets(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Ошибка загрузки датасетов');
      } finally {
        setLoading(false);
      }
    };
    fetchDatasets();
  }, []);

  // Функции для управления источниками
  const handleAddSource = () => {
    setNewDataset(prev => ({
      ...prev,
      sources: [...prev.sources, { type: 'kaggle', url: '', description: '' }]
    }));
  };

  const handleRemoveSource = (index: number) => {
    if (newDataset.sources.length <= 1) return;
    setNewDataset(prev => ({
      ...prev,
      sources: prev.sources.filter((_, i) => i !== index)
    }));
  };

  const handleSourceChange = (index: number, field: keyof SourceItem, value: string) => {
    setNewDataset(prev => {
      const updatedSources = [...prev.sources];
      updatedSources[index] = { ...updatedSources[index], [field]: value };
      return { ...prev, sources: updatedSources };
    });
  };

  // Функции для управления классами
  const handleAddClass = () => {
    const trimmed = newClassName.trim();
    if (!trimmed) {
      alert('Введите название класса');
      return;
    }
    if (newDataset.class_names.includes(trimmed)) {
      alert('Класс с таким названием уже существует');
      return;
    }
    setNewDataset(prev => ({
      ...prev,
      class_names: [...prev.class_names, trimmed]
    }));
    setNewClassName('');
    classInputRef.current?.focus();
  };

  const handleRemoveClass = (className: string) => {
    setNewDataset(prev => ({
      ...prev,
      class_names: prev.class_names.filter(c => c !== className)
    }));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddClass();
    }
  };

  const handleNewDatasetChange = (field: keyof Omit<typeof newDataset, 'file' | 'sources' | 'class_names'>, value: any) => {
    setNewDataset(prev => ({ ...prev, [field]: value }));
  };

  const handleVersionChange = (field: keyof NewVersion, value: string) => {
    setNewDataset(prev => ({
      ...prev,
      version: { ...prev.version, [field]: value }
    }));
  };

  const handleCreateDataset = async () => {
    if (!newDataset.dataset_id || !newDataset.name || newDataset.class_names.length === 0 || !newDataset.version.version_id) {
      alert('Заполните обязательные поля: ID, название, классы, ID версии');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      if (newDataset.file) {
        const uploaded = await datasetService.uploadFile(newDataset.dataset_id, newDataset.file);
        if (!uploaded) throw new Error('Не удалось загрузить файл');
      }

      const created = await datasetService.createDataset({
        dataset_id: newDataset.dataset_id,
        name: newDataset.name,
        description: newDataset.description,
        class_names: newDataset.class_names,
        sources: newDataset.sources,
        type: newDataset.type,
        task: newDataset.task,
        version: newDataset.version,
      });

      if (!created) throw new Error('Не удалось создать датасет');

      const updated = await datasetService.getDatasets();
      setDatasets(updated);

      setNewDataset({
        dataset_id: '',
        name: '',
        description: '',
        class_names: [],
        sources: [{ type: 'kaggle', url: '', description: '' }],
        type: 'image',
        task: 'classification',
        version: { version_id: '', description: '' },
        file: null,
      });
      setShowAddForm(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка создания датасета');
    } finally {
      setLoading(false);
    }
  };

  const handleOpenVersionForm = (datasetId: string) => {
    setShowVersionForm(datasetId);
    setNewVersion({ version_id: '', description: '', file: null });
  };

  const handleAddVersion = async (datasetId: string) => {
    if (!newVersion.version_id) {
      alert('Введите ID версии');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      if (newVersion.file) {
        const uploaded = await datasetService.uploadFile(newVersion.version_id, newVersion.file);
        if (!uploaded) throw new Error('Не удалось загрузить файл');
      }

      const created = await datasetService.createVersion(datasetId, {
        version_id: newVersion.version_id,
        description: newVersion.description,
      });

      if (!created) throw new Error('Не удалось создать версию');

      const updated = await datasetService.getDatasets();
      setDatasets(updated);

      setShowVersionForm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка добавления версии');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDataset = async (id: string) => {
    if (!window.confirm('Вы уверены, что хотите удалить датасет?')) return;

    try {
      setLoading(true);
      setError(null);
      const deleted = await datasetService.deleteDataset(id);
      if (!deleted) throw new Error('Не удалось удалить датасет');
      setDatasets(prev => prev.filter(ds => ds.dataset_id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка удаления датасета');
    } finally {
      setLoading(false);
    }
  };

  if (loading && datasets.length === 0) {
    return <div className="loading">Загрузка датасетов...</div>;
  }

  return (
    <div className="datasets-page">
      <div className="datasets-header">
        <h1>Управление датасетами</h1>
        <p className="datasets-description">
          Загружайте, удаляйте и управляйте версиями датасетов для классификации изображений.
        </p>
        {error && <div className="error-message">{error}</div>}
        {!showAddForm && (
          <button className="button" onClick={() => setShowAddForm(true)} disabled={loading}>
            <i className="fas fa-plus"></i> Новый датасет
          </button>
        )}
      </div>

      {showAddForm && (
        <DatasetForm
          newDataset={newDataset}
          newClassName={newClassName}
          classInputRef={classInputRef}
          loading={loading}
          onNewDatasetChange={handleNewDatasetChange}
          onVersionChange={handleVersionChange}
          onAddClass={handleAddClass}
          onRemoveClass={handleRemoveClass}
          onClassInputChange={setNewClassName}
          onKeyPress={handleKeyPress}
          onAddSource={handleAddSource}
          onRemoveSource={handleRemoveSource}
          onSourceChange={handleSourceChange}
          onFileSelect={(file) => setNewDataset(prev => ({ ...prev, file }))}
          onSubmit={handleCreateDataset}
          onCancel={() => setShowAddForm(false)}
        />
      )}

      <div className="datasets-list">
        {datasets.length === 0 && !loading ? (
          <p className="no-data">Пока нет ни одного датасета. Создайте первый!</p>
        ) : (
          datasets.map(dataset => (
            <DatasetCard
              key={dataset.dataset_id}
              dataset={dataset}
              loading={loading}
              showVersionForm={showVersionForm === dataset.dataset_id}
              onAddVersion={() => handleOpenVersionForm(dataset.dataset_id)}
              onDelete={() => handleDeleteDataset(dataset.dataset_id)}
              versionForm={
                showVersionForm === dataset.dataset_id && (
                  <AddVersionForm
                    version={newVersion}
                    loading={loading}
                    onVersionChange={setNewVersion}
                    onSubmit={() => handleAddVersion(dataset.dataset_id)}
                    onCancel={() => setShowVersionForm(null)}
                  />
                )
              }
            />
          ))
        )}
      </div>
    </div>
  );
};

export default Datasets;