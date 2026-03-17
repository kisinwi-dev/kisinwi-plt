import React, { useRef } from 'react';
import FileUploader from '../FileUploader';
import type { NewDataset, NewVersion, SourceItem } from '../../types/dataset';

interface DatasetFormProps {
  newDataset: NewDataset & { file: File | null };
  newClassName: string;
  classInputRef: React.RefObject<HTMLInputElement | null>; // допускаем null
  onNewDatasetChange: (field: keyof Omit<NewDataset & { file: File | null }, 'file' | 'sources' | 'class_names'>, value: any) => void;
  loading: boolean;
  onVersionChange: (field: keyof NewVersion, value: string) => void;
  onAddClass: () => void;
  onRemoveClass: (className: string) => void;
  onClassInputChange: (value: string) => void;
  onKeyPress: (e: React.KeyboardEvent) => void;
  onAddSource: () => void;
  onRemoveSource: (index: number) => void;
  onSourceChange: (index: number, field: keyof SourceItem, value: string) => void;
  onFileSelect: (file: File | null) => void;
  onSubmit: () => void;
  onCancel: () => void;
}

const DatasetForm: React.FC<DatasetFormProps> = ({
  newDataset,
  newClassName,
  classInputRef,
  loading,
  onNewDatasetChange,
  onVersionChange,
  onAddClass,
  onRemoveClass,
  onClassInputChange,
  onKeyPress,
  onAddSource,
  onRemoveSource,
  onSourceChange,
  onFileSelect,
  onSubmit,
  onCancel,
}) => {
  return (
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
              onChange={(e) => onNewDatasetChange('dataset_id', e.target.value)}
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
              onChange={(e) => onNewDatasetChange('name', e.target.value)}
              disabled={loading}
            />
          </div>
          <div className="form-field full-width">
            <label htmlFor="dataset-description">Описание</label>
            <textarea
              id="dataset-description"
              placeholder="Краткое описание датасета"
              value={newDataset.description}
              onChange={(e) => onNewDatasetChange('description', e.target.value)}
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
              onChange={(e) => onClassInputChange(e.target.value)}
              onKeyPress={onKeyPress}
              disabled={loading}
            />
            <button type="button" className="add-class-btn" onClick={onAddClass} disabled={loading}>
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
                    onClick={() => onRemoveClass(className)}
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
                    onChange={(e) => onSourceChange(index, 'type', e.target.value)}
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
                    onClick={() => onRemoveSource(index)}
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
                    onChange={(e) => onSourceChange(index, 'url', e.target.value)}
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
                    onChange={(e) => onSourceChange(index, 'description', e.target.value)}
                    className="source-input"
                    disabled={loading}
                  />
                </div>
              </div>
            </div>
          ))}
          <button type="button" className="add-source-btn" onClick={onAddSource} disabled={loading}>
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
              onChange={(e) => onNewDatasetChange('type', e.target.value)}
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
              onChange={(e) => onNewDatasetChange('task', e.target.value)}
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
              onChange={(e) => onVersionChange('version_id', e.target.value)}
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
              onChange={(e) => onVersionChange('description', e.target.value)}
              disabled={loading}
            />
          </div>
        </div>
      </div>

      {/* Файл датасета */}
      <div className="form-section">
        <h3>Файл датасета</h3>
        <FileUploader
          onFileSelect={onFileSelect}
          accept=".zip"
          currentFile={newDataset.file}
        />
        <span className="field-hint">Загрузите архив с данными</span>
      </div>

      <div className="form-actions">
        <button className="button" onClick={onSubmit} disabled={loading}>
          {loading ? 'Создание...' : 'Создать датасет'}
        </button>
        <button className="button secondary" onClick={onCancel} disabled={loading}>
          Отмена
        </button>
      </div>
    </div>
  );
};

export default DatasetForm;