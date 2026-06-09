import React from 'react';
import FileUploader from '../FileUploader';
import SourcesEditor from './SourcesEditor';
import Select from '../common/Select';
import type { NewDataset, NewVersion, SourceItem } from '../../types/dataset';

const TYPE_OPTIONS = [
  { value: 'image', label: 'Image' },
  { value: 'text', label: 'Text' },
  { value: 'tabular', label: 'Tabular' },
  { value: 'other', label: 'Other' },
];

const TASK_OPTIONS = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'detection', label: 'Detection' },
  { value: 'segmentation', label: 'Segmentation' },
  { value: 'other', label: 'Other' },
];

type VersionFormData = Omit<NewVersion, 'id_data'>;

interface DatasetFormProps {
  newDataset: Omit<NewDataset, 'version'> & { version: VersionFormData; file: File | null };
  loading: boolean;
  onNewDatasetChange: (field: keyof Omit<NewDataset, 'version'>, value: string) => void;
  onVersionChange: (field: keyof VersionFormData, value: string) => void;
  onVersionSourcesChange: (sources: SourceItem[]) => void;
  onFileSelect: (file: File | null) => void;
  onSubmit: () => void;
  onCancel: () => void;
}

const DatasetForm: React.FC<DatasetFormProps> = ({
  newDataset,
  loading,
  onNewDatasetChange,
  onVersionChange,
  onVersionSourcesChange,
  onFileSelect,
  onSubmit,
  onCancel,
}) => {
  return (
    <div className="add-dataset-form">
      <div className="add-dataset-form-head">
        <h2>Создать новый датасет</h2>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className="fas fa-circle-info"></i> Основная информация</h3>
          <p className="form-section-hint">Как датасет будет называться и о чём он.</p>
        </div>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="dataset-name">Название <span className="required-star">*</span></label>
            <input
              id="dataset-name"
              type="text"
              autoComplete="off"
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
              autoComplete="off"
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
        <div className="form-section-head">
          <h3><i className="fas fa-shapes"></i> Тип и задача</h3>
          <p className="form-section-hint">Какого вида данные и для чего модель.</p>
        </div>
        <div className="form-row">
          <div className="form-field">
            <label htmlFor="dataset-type">Тип данных</label>
            <Select
              icon="fas fa-shapes"
              ariaLabel="Тип данных"
              value={newDataset.type}
              options={TYPE_OPTIONS}
              onChange={(v) => onNewDatasetChange('type', v)}
            />
          </div>
          <div className="form-field">
            <label htmlFor="dataset-task">Задача</label>
            <Select
              icon="fas fa-bullseye"
              ariaLabel="Задача"
              value={newDataset.task}
              options={TASK_OPTIONS}
              onChange={(v) => onNewDatasetChange('task', v)}
            />
          </div>
        </div>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className="fas fa-code-branch"></i> Начальная версия <span className="required-star">*</span></h3>
          <p className="form-section-hint">Первая версия данных и откуда они взяты.</p>
        </div>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="version-name">Название версии <span className="required-star">*</span></label>
            <input
              id="version-name"
              type="text"
              autoComplete="off"
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
              autoComplete="off"
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
          onChange={onVersionSourcesChange}
        />
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className="fas fa-file-zipper"></i> Файл датасета</h3>
          <p className="form-section-hint">Архив с изображениями, разложенными по классам.</p>
        </div>
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
