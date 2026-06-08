import React, { useState, useEffect } from 'react';
import { datasetService } from '../services/datasetService';
import type { Dataset, NewDataset, SourceItem, VersionSplitsResponse } from '../types/dataset';
import FileUploader from '../components/FileUploader';
import './Datasets.css';
import { useNotification } from '../contexts/NotificationContext';
import { formatBytes, formatDateTime } from '../utils/format';
import VersionSplitsStats from '../components/datasets/VersionSplitsStats';

const EMPTY_SOURCE = (): SourceItem => ({ type: 'kaggle', url: null, description: '' });

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
  const [versionStats, setVersionStats] = useState<Record<string, VersionSplitsResponse | 'loading' | 'error'>>({});

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

  const handleVersionFieldChange = (field: 'name' | 'description', value: string) => {
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

  // ── Handlers: new version form ─────────────────────────────────────────────

  const handleNewVersionFieldChange = (field: 'name' | 'description', value: string) => {
    setNewVersion(prev => ({ ...prev, [field]: value }));
  };

  const handleNewVersionSourceAdd = () => {
    setNewVersion(prev => ({ ...prev, sources: [...prev.sources, EMPTY_SOURCE()] }));
  };

  const handleNewVersionSourceRemove = (index: number) => {
    if (newVersion.sources.length <= 1) return;
    setNewVersion(prev => ({ ...prev, sources: prev.sources.filter((_, i) => i !== index) }));
  };

  const handleNewVersionSourceChange = (index: number, field: keyof SourceItem, value: string) => {
    setNewVersion(prev => {
      const sources = [...prev.sources];
      sources[index] = { ...sources[index], [field]: field === 'url' ? (value || null) : value };
      return { ...prev, sources };
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
      const id_data = `upload_${Date.now()}`;

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
      const id_data = `upload_${Date.now()}`;

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

  const handleShowVersionStats = async (datasetId: string, versionId: string) => {
    const key = `${datasetId}:${versionId}`;
    if (versionStats[key]) {
      setVersionStats(prev => { const next = { ...prev }; delete next[key]; return next; });
      return;
    }
    setVersionStats(prev => ({ ...prev, [key]: 'loading' }));
    try {
      const data = await datasetService.getVersionSplits(datasetId, versionId);
      setVersionStats(prev => ({ ...prev, [key]: data }));
    } catch {
      setVersionStats(prev => ({ ...prev, [key]: 'error' }));
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

  // ── Render helpers ─────────────────────────────────────────────────────────

  const renderSourceFields = (
    sources: SourceItem[],
    onAdd: () => void,
    onRemove: (i: number) => void,
    onChange: (i: number, field: keyof SourceItem, value: string) => void,
  ) => (
    <div className="sources-container">
      {sources.map((source, index) => (
        <div key={index} className="source-card">
          <div className="source-header">
            <div className="source-type-wrapper">
              <select
                value={source.type}
                onChange={(e) => onChange(index, 'type', e.target.value)}
                className="source-type-select"
                disabled={loading}
              >
                <option value="kaggle">📊 Kaggle</option>
                <option value="url">🌐 URL</option>
                <option value="huggingface">🤗 Hugging Face</option>
                <option value="other">📁 Другой</option>
              </select>
            </div>
            {sources.length > 1 && (
              <button
                type="button"
                className="source-remove-btn"
                onClick={() => onRemove(index)}
                title="Удалить источник"
                disabled={loading}
              >
                <i className="fas fa-trash-alt"></i>
              </button>
            )}
          </div>
          <div className="source-fields">
            <div className="form-field">
              <label>URL источника</label>
              <input
                type="url"
                placeholder="https://..."
                value={source.url ?? ''}
                onChange={(e) => onChange(index, 'url', e.target.value)}
                className="source-input"
                disabled={loading}
              />
            </div>
            <div className="form-field">
              <label>Описание</label>
              <input
                type="text"
                placeholder="например: оригинальный источник"
                value={source.description}
                onChange={(e) => onChange(index, 'description', e.target.value)}
                className="source-input"
                disabled={loading}
              />
            </div>
          </div>
        </div>
      ))}
      <button type="button" className="add-source-btn" onClick={onAdd} disabled={loading}>
        <i className="fas fa-plus-circle"></i> Добавить ещё источник
      </button>
    </div>
  );

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

      {/* Форма создания нового датасета */}
      {showAddForm && (
        <div className="add-dataset-form">
          <h2>Создать новый датасет</h2>

          <div className="form-section">
            <h3>Основная информация</h3>
            <div className="form-grid">
              <div className="form-field">
                <label htmlFor="dataset-name">Название <span className="required-star">*</span></label>
                <input
                  id="dataset-name"
                  type="text"
                  placeholder="например: CIFAR-10"
                  value={newDataset.name}
                  onChange={(e) => handleNewDatasetChange('name', e.target.value)}
                  disabled={loading}
                />
              </div>
              <div className="form-field full-width">
                <label htmlFor="dataset-description">Описание</label>
                <textarea
                  id="dataset-description"
                  placeholder="Краткое описание датасета"
                  value={newDataset.description}
                  onChange={(e) => handleNewDatasetChange('description', e.target.value)}
                  rows={3}
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          {/* Тип и задача */}
          <div className="form-section">
            <h3>Тип и задача</h3>
            <div className="form-row">
              <div className="form-field">
                <label htmlFor="dataset-type">Тип данных</label>
                <select
                  id="dataset-type"
                  value={newDataset.type}
                  onChange={(e) => handleNewDatasetChange('type', e.target.value)}
                  disabled={loading}
                >
                  <option value="image">Image</option>
                  <option value="text">Text</option>
                  <option value="tabular">Tabular</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div className="form-field">
                <label htmlFor="dataset-task">Задача</label>
                <select
                  id="dataset-task"
                  value={newDataset.task}
                  onChange={(e) => handleNewDatasetChange('task', e.target.value)}
                  disabled={loading}
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                  <option value="detection">Detection</option>
                  <option value="segmentation">Segmentation</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>
          </div>

          {/* Начальная версия */}
          <div className="form-section">
            <h3>Начальная версия <span className="required-star">*</span></h3>
            <div className="form-grid">
              <div className="form-field">
                <label htmlFor="version-name">Название версии <span className="required-star">*</span></label>
                <input
                  id="version-name"
                  type="text"
                  placeholder="например: v1.0"
                  value={newDataset.version.name}
                  onChange={(e) => handleVersionFieldChange('name', e.target.value)}
                  disabled={loading}
                />
              </div>
              <div className="form-field">
                <label htmlFor="version-description">Описание версии</label>
                <input
                  id="version-description"
                  type="text"
                  placeholder="Краткое описание"
                  value={newDataset.version.description}
                  onChange={(e) => handleVersionFieldChange('description', e.target.value)}
                  disabled={loading}
                />
              </div>
            </div>

            <h4>Источники данных</h4>
            {renderSourceFields(
              newDataset.version.sources,
              handleVersionSourceAdd,
              handleVersionSourceRemove,
              handleVersionSourceChange,
            )}
          </div>

          {/* Файл датасета */}
          <div className="form-section">
            <h3>Файл датасета</h3>
            <FileUploader
              onFileSelect={(file) => setNewDataset(prev => ({ ...prev, file }))}
              accept=".zip"
              currentFile={newDataset.file}
            />
            <span className="field-hint">Загрузите архив с данными. Классы определяются автоматически по структуре папок.</span>
          </div>

          <div className="form-actions">
            <button className="button" onClick={handleCreateDataset} disabled={loading}>
              {loading ? 'Создание...' : 'Создать датасет'}
            </button>
            <button className="button secondary" onClick={() => setShowAddForm(false)} disabled={loading}>
              Отмена
            </button>
          </div>
        </div>
      )}

      {/* Список датасетов */}
      <div className="datasets-list">
        {datasets.length === 0 && !loading ? (
          <p className="empty-state">Пока нет ни одного датасета. Создайте первый!</p>
        ) : (
          datasets.map(dataset => (
            <div key={dataset.id} className="card dataset-card">
              <div className="dataset-header">
                <h2>{dataset.name}</h2>
                <div className="dataset-actions">
                  <button
                    className="icon-button"
                    onClick={() => handleOpenVersionForm(dataset.id)}
                    title="Добавить версию"
                    disabled={loading}
                  >
                    <i className="fas fa-code-branch"></i>
                  </button>
                  <button
                    className="icon-button"
                    onClick={() => handleDeleteDataset(dataset.id)}
                    title="Удалить датасет"
                    disabled={loading}
                  >
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
              </div>

              <div className="dataset-meta">
                <span><i className="fas fa-tag"></i> {dataset.type} / {dataset.task}</span>
                <span><i className="fas fa-calendar-alt"></i> Создан: {formatDateTime(dataset.created_at)}</span>
                <span><i className="fas fa-sync-alt"></i> Обновлён: {formatDateTime(dataset.updated_at)}</span>
              </div>

              <p className="dataset-description">{dataset.description}</p>

              {dataset.classes_names.length > 0 && (
                <div className="dataset-classes">
                  <h4>Классы ({dataset.classes_count})</h4>
                  <div className="tag-list">
                    {dataset.classes_names.slice(0, 10).map(className => (
                      <span key={className} className="tag">{className}</span>
                    ))}
                    {dataset.classes_names.length > 10 && <span className="tag">...</span>}
                  </div>
                </div>
              )}

              {/* Форма добавления версии */}
              {showVersionForm === dataset.id && (
                <div className="add-version-form">
                  <h5>Добавить версию</h5>
                  <input
                    type="text"
                    placeholder="Название версии *"
                    value={newVersion.name}
                    onChange={(e) => handleNewVersionFieldChange('name', e.target.value)}
                    disabled={loading}
                  />
                  <input
                    type="text"
                    placeholder="Описание"
                    value={newVersion.description}
                    onChange={(e) => handleNewVersionFieldChange('description', e.target.value)}
                    disabled={loading}
                  />
                  <h5>Источники</h5>
                  {renderSourceFields(
                    newVersion.sources,
                    handleNewVersionSourceAdd,
                    handleNewVersionSourceRemove,
                    handleNewVersionSourceChange,
                  )}
                  <h5>Файл версии</h5>
                  <FileUploader
                    onFileSelect={(file) => setNewVersion(prev => ({ ...prev, file }))}
                    accept=".zip,.tar,.gz,.rar,.7z"
                    currentFile={newVersion.file}
                  />
                  <div className="form-actions">
                    <button className="button small" onClick={() => handleAddVersion(dataset.id)} disabled={loading}>
                      {loading ? 'Сохранение...' : 'Сохранить'}
                    </button>
                    <button className="button small secondary" onClick={() => setShowVersionForm(null)} disabled={loading}>
                      Отмена
                    </button>
                  </div>
                </div>
              )}

              {/* Версии */}
              <div className="versions-section">
                <h4>Версии ({dataset.versions.length})</h4>
                {dataset.versions.length === 0 ? (
                  <p className="no-versions">Нет версий</p>
                ) : (
                  <div className="versions-list">
                    {dataset.versions.map(ver => (
                      <div key={ver.id} className="version-item">
                        <div className="version-header">
                          <span className="version-name">
                            {ver.name}
                            {ver.id === dataset.default_version_id && (
                              <span className="default-badge-inline"> (по умолчанию)</span>
                            )}
                          </span>
                          <div className="version-actions">
                            <span className="version-size">{formatBytes(ver.size_bytes)}</span>
                            <button
                              className="icon-button small"
                              onClick={() => handleShowVersionStats(dataset.id, ver.id)}
                              title="Статистика"
                            >
                              <i className="fas fa-chart-bar"></i>
                            </button>
                            <button
                              className="icon-button small"
                              onClick={() => handleDeleteVersion(dataset.id, ver.id)}
                              title="Удалить версию"
                              disabled={loading}
                            >
                              <i className="fas fa-trash"></i>
                            </button>
                          </div>
                        </div>
                        <span className="version-date">Дата загрузки: {formatDateTime(ver.created_at)}</span>
                        <p className="version-description">Описание: {ver.description}</p>
                        <div className="version-stats">
                          <span>Всего: {ver.num_samples.toLocaleString()}</span>
                        </div>
                        {ver.sources.length > 0 && (
                          <div className="dataset-sources">
                            {ver.sources.map((src, idx) => (
                              <div key={idx} className="source-item">
                                {src.url ? (
                                  <a href={src.url} target="_blank" rel="noopener noreferrer" className="source-type-badge source-type-badge--link">
                                    {src.type}
                                  </a>
                                ) : (
                                  <span className="source-type-badge">{src.type}</span>
                                )}
                                {src.description && <span className="source-description">{src.description}</span>}
                              </div>
                            ))}
                          </div>
                        )}
                        {versionStats[`${dataset.id}:${ver.id}`] === 'loading' && (
                          <p className="stats-loading">Загрузка статистики...</p>
                        )}
                        {versionStats[`${dataset.id}:${ver.id}`] === 'error' && (
                          <p className="stats-error">Не удалось загрузить статистику</p>
                        )}
                        {typeof versionStats[`${dataset.id}:${ver.id}`] === 'object' && (
                          <VersionSplitsStats
                            stats={versionStats[`${dataset.id}:${ver.id}`] as VersionSplitsResponse}
                            onClose={() => handleShowVersionStats(dataset.id, ver.id)}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Datasets;
