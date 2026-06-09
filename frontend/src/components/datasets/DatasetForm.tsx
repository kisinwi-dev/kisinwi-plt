import React from 'react';
import FileUploader from '../FileUploader';
import SourcesEditor from './SourcesEditor';
import type { NewDataset, NewVersion, SourceItem } from '../../types/dataset';

type VersionFormData = Omit<NewVersion, 'id_data'>;

interface DatasetFormProps {
  newDataset: Omit<NewDataset, 'version'> & { version: VersionFormData; file: File | null };
  loading: boolean;
  onNewDatasetChange: (field: keyof Omit<NewDataset, 'version'>, value: string) => void;
  onVersionChange: (field: keyof VersionFormData, value: string) => void;
  onVersionSourceAdd: () => void;
  onVersionSourceRemove: (index: number) => void;
  onVersionSourceChange: (index: number, field: keyof SourceItem, value: string) => void;
  onFileSelect: (file: File | null) => void;
  onSubmit: () => void;
  onCancel: () => void;
}

const DatasetForm: React.FC<DatasetFormProps> = ({
  newDataset,
  loading,
  onNewDatasetChange,
  onVersionChange,
  onVersionSourceAdd,
  onVersionSourceRemove,
  onVersionSourceChange,
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
              onChange={(e) => onVersionChange('name', e.target.value)}
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

        <h4>Источники данных</h4>
        <SourcesEditor
          sources={newDataset.version.sources}
          loading={loading}
          onAdd={onVersionSourceAdd}
          onRemove={onVersionSourceRemove}
          onChange={onVersionSourceChange}
        />
      </div>

      <div className="form-section">
        <h3>Файл датасета</h3>
        <FileUploader
          onFileSelect={onFileSelect}
          accept=".zip"
          currentFile={newDataset.file}
        />
        <span className="field-hint">Загрузите архив с данными. Классы определяются автоматически.</span>
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
