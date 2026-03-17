import React, { useState, useRef, useEffect } from 'react';
import { datasetService } from '../services/datasetService';
import type { Dataset, NewDataset, NewVersion, SourceItem, Version } from '../types/dataset';
import FileUploader from '../components/FileUploader';
import './Datasets.css';

const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

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

  // Загрузка датасетов при монтировании
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

  // Создание датасета (сначала файл, потом метаданные)
  const handleCreateDataset = async () => {
    if (!newDataset.dataset_id || !newDataset.name || newDataset.class_names.length === 0 || !newDataset.version.version_id) {
      alert('Заполните обязательные поля: ID, название, классы, ID версии');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // 1. Загружаем файл (если есть)
      if (newDataset.file) {
        const uploaded = await datasetService.uploadFile(newDataset.dataset_id, newDataset.file);
        if (!uploaded) throw new Error('Не удалось загрузить файл');
      }

      // 2. Создаём метаданные датасета
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

      // 3. Обновляем список датасетов
      const updated = await datasetService.getDatasets();
      setDatasets(updated);

      // Сброс формы
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

  // Открыть форму добавления версии
  const handleOpenVersionForm = (datasetId: string) => {
    setShowVersionForm(datasetId);
    setNewVersion({ version_id: '', description: '', file: null });
  };

  // Добавление новой версии (сначала файл, потом метаданные)
  const handleAddVersion = async (datasetId: string) => {
    if (!newVersion.version_id) {
      alert('Введите ID версии');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // 1. Загружаем файл (если есть)
      if (newVersion.file) {
        const uploaded = await datasetService.uploadFile(newVersion.version_id, newVersion.file);
        if (!uploaded) throw new Error('Не удалось загрузить файл');
      }

      // 2. Создаём метаданные версии
      const created = await datasetService.createVersion(datasetId, {
        version_id: newVersion.version_id,
        description: newVersion.description,
      });

      if (!created) throw new Error('Не удалось создать версию');

      // 3. Обновляем список датасетов (или конкретный датасет)
      const updated = await datasetService.getDatasets();
      setDatasets(updated);

      setShowVersionForm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка добавления версии');
    } finally {
      setLoading(false);
    }
  };

  // Удаление датасета
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

      {/* Форма создания нового датасета */}
      {showAddForm && (
        <div className="add-dataset-form">
          <h2>Создать новый датасет</h2>

          <div className="form-section">
            <h3>Основная информация</h3>
            <div className="form-grid">
              <div className="form-field">
                <label htmlFor="dataset-id">ID датасета <span className="required-star">*</span></label>
                <input
                  id="dataset-id"
                  type="text"
                  placeholder="например: cifar10"
                  value={newDataset.dataset_id}
                  onChange={(e) => handleNewDatasetChange('dataset_id', e.target.value)}
                  disabled={loading}
                />
                <span className="field-hint">Уникальный идентификатор, только латиница и цифры</span>
              </div>
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

          {/* Блок классов */}
          <div className="form-section">
            <h3>Классы <span className="required-star">*</span></h3>
            <div className="classes-section">
              <div className="class-input-group">
                <input
                  ref={classInputRef}
                  type="text"
                  placeholder="Введите название класса"
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={loading}
                />
                <button type="button" className="add-class-btn" onClick={handleAddClass} disabled={loading}>
                  <i className="fas fa-plus-circle"></i> Добавить класс
                </button>
              </div>
              <span className="field-hint">Каждый класс добавляется отдельно, можно использовать Enter</span>

              {newDataset.class_names.length > 0 ? (
                <div className="class-tags-container">
                  {newDataset.class_names.map((className) => (
                    <span key={className} className="class-tag">
                      {className}
                      <button
                        type="button"
                        className="remove-class-btn"
                        onClick={() => handleRemoveClass(className)}
                        title="Удалить класс"
                        disabled={loading}
                      >
                        <i className="fas fa-times"></i>
                      </button>
                    </span>
                  ))}
                </div>
              ) : (
                <p className="field-error">Добавьте хотя бы один класс</p>
              )}
            </div>
          </div>

          {/* Источники данных */}
          <div className="form-section">
            <h3>Источники данных <span className="required-star">*</span></h3>
            <div className="sources-container">
              {newDataset.sources.map((source, index) => (
                <div key={index} className="source-card">
                  <div className="source-header">
                    <div className="source-type-wrapper">
                      <select
                        value={source.type}
                        onChange={(e) => handleSourceChange(index, 'type', e.target.value)}
                        className="source-type-select"
                        disabled={loading}
                      >
                        <option value="kaggle">📊 Kaggle</option>
                        <option value="url">🌐 URL</option>
                        <option value="huggingface">🤗 Hugging Face</option>
                        <option value="other">📁 Другой</option>
                      </select>
                    </div>
                    {newDataset.sources.length > 1 && (
                      <button
                        type="button"
                        className="source-remove-btn"
                        onClick={() => handleRemoveSource(index)}
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
                        value={source.url}
                        onChange={(e) => handleSourceChange(index, 'url', e.target.value)}
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
                        onChange={(e) => handleSourceChange(index, 'description', e.target.value)}
                        className="source-input"
                        disabled={loading}
                      />
                    </div>
                  </div>
                </div>
              ))}
              <button type="button" className="add-source-btn" onClick={handleAddSource} disabled={loading}>
                <i className="fas fa-plus-circle"></i> Добавить ещё источник
              </button>
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
                <label htmlFor="version-id">ID версии</label>
                <input
                  id="version-id"
                  type="text"
                  placeholder="например: v1.0"
                  value={newDataset.version.version_id}
                  onChange={(e) => handleVersionChange('version_id', e.target.value)}
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
                  onChange={(e) => handleVersionChange('description', e.target.value)}
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          {/* Файл датасета */}
          <div className="form-section">
            <h3>Файл датасета</h3>
            <FileUploader
              onFileSelect={(file) => setNewDataset(prev => ({ ...prev, file }))}
              accept=".zip"
              currentFile={newDataset.file}
            />
            <span className="field-hint">Загрузите архив с данными</span>
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
          <p className="no-data">Пока нет ни одного датасета. Создайте первый!</p>
        ) : (
          datasets.map(dataset => (
            <div key={dataset.dataset_id} className="dataset-card">
              <div className="dataset-header">
                <h2>{dataset.name}</h2>
                <div className="dataset-actions">
                  <button
                    className="icon-button"
                    onClick={() => handleOpenVersionForm(dataset.dataset_id)}
                    title="Добавить версию"
                    disabled={loading}
                  >
                    <i className="fas fa-code-branch"></i>
                  </button>
                  <button
                    className="icon-button"
                    onClick={() => handleDeleteDataset(dataset.dataset_id)}
                    title="Удалить датасет"
                    disabled={loading}
                  >
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
              </div>

              <p className="dataset-description">{dataset.description}</p>

              <div className="dataset-meta">
                <span><i className="fas fa-tag"></i> {dataset.type} / {dataset.task}</span>
                <span><i className="fas fa-calendar-alt"></i> Создан: {new Date(dataset.created_at).toLocaleDateString()}</span>
                <span><i className="fas fa-sync-alt"></i> Обновлён: {new Date(dataset.updated_at).toLocaleDateString()}</span>
              </div>

              {/* Источники */}
              {dataset.sources && dataset.sources.length > 0 && (
                <div className="dataset-sources">
                  <h4>Источники</h4>
                  {dataset.sources.map((src, idx) => (
                    <div key={idx} className="source-item">
                      <span className="source-type-badge">{src.type}</span>
                      <a href={src.url} target="_blank" rel="noopener noreferrer" className="source-link">
                        {src.url}
                      </a>
                      {src.description && (
                        <span className="source-description">{src.description}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              <div className="dataset-classes">
                <h4>Классы ({dataset.num_classes})</h4>
                <div className="class-tags">
                  {dataset.class_names.slice(0, 10).map(className => (
                    <span key={className} className="class-tag">{className}</span>
                  ))}
                  {dataset.class_names.length > 10 && <span className="class-tag">...</span>}
                </div>
              </div>

              {/* Форма добавления версии */}
              {showVersionForm === dataset.dataset_id && (
                <div className="add-version-form">
                  <h5>Добавить версию</h5>
                  <input
                    type="text"
                    placeholder="ID версии *"
                    value={newVersion.version_id}
                    onChange={(e) => setNewVersion({ ...newVersion, version_id: e.target.value })}
                    disabled={loading}
                  />
                  <input
                    type="text"
                    placeholder="Описание"
                    value={newVersion.description}
                    onChange={(e) => setNewVersion({ ...newVersion, description: e.target.value })}
                    disabled={loading}
                  />
                  <h5>Файл версии</h5>
                  <FileUploader
                    onFileSelect={(file) => setNewVersion(prev => ({ ...prev, file }))}
                    accept=".zip,.tar,.gz,.rar,.7z"
                    currentFile={newVersion.file}
                  />
                  <div className="form-actions">
                    <button className="button small" onClick={() => handleAddVersion(dataset.dataset_id)} disabled={loading}>
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
                      <div key={ver.version_id} className="version-item">
                        <div className="version-header">
                          <span className="version-name">
                            {ver.version_id}
                            {ver.version_id === dataset.default_version_id && (
                              <span className="default-badge-inline"> (по умолчанию)</span>
                            )}
                          </span>
                          <span className="version-size">{formatBytes(ver.size_bytes)}</span>
                        </div>
                        <span className="version-date">{'Дата загрузки: ' + new Date(ver.created_at).toLocaleDateString()}</span>
                        <p className="version-description">{'Описание: ' + ver.description}</p>
                        <div className="version-stats">
                          <span>Всего: {ver.num_samples.toLocaleString()}</span>
                          <span>Train: {ver.num_train.toLocaleString()}</span>
                          {ver.num_val > 0 && <span>Val: {ver.num_val.toLocaleString()}</span>}
                          <span>Test: {ver.num_test.toLocaleString()}</span>
                        </div>
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