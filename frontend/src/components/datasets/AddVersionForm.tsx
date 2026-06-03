import React from 'react';
import FileUploader from '../FileUploader';
import type { Source, SourceItem } from '../../types/dataset';

interface VersionFormState {
  name: string;
  description: string;
  sources: Source;
  file: File | null;
}

interface AddVersionFormProps {
  version: VersionFormState;
  loading: boolean;
  onVersionChange: (version: VersionFormState) => void;
  onSubmit: () => void;
  onCancel: () => void;
}

const AddVersionForm: React.FC<AddVersionFormProps> = ({
  version,
  loading,
  onVersionChange,
  onSubmit,
  onCancel,
}) => {
  const handleFieldChange = (field: 'name' | 'description', value: string) => {
    onVersionChange({ ...version, [field]: value });
  };

  const handleSourceChange = (index: number, field: keyof SourceItem, value: string) => {
    const sources = [...version.sources];
    sources[index] = { ...sources[index], [field]: field === 'url' ? (value || null) : value };
    onVersionChange({ ...version, sources });
  };

  const handleSourceAdd = () => {
    onVersionChange({
      ...version,
      sources: [...version.sources, { type: 'kaggle', url: null, description: '' }],
    });
  };

  const handleSourceRemove = (index: number) => {
    if (version.sources.length <= 1) return;
    onVersionChange({ ...version, sources: version.sources.filter((_, i) => i !== index) });
  };

  return (
    <div className="add-version-form">
      <h5>Добавить версию</h5>
      <input
        type="text"
        placeholder="Название версии *"
        value={version.name}
        onChange={(e) => handleFieldChange('name', e.target.value)}
        disabled={loading}
      />
      <input
        type="text"
        placeholder="Описание"
        value={version.description}
        onChange={(e) => handleFieldChange('description', e.target.value)}
        disabled={loading}
      />

      <h5>Источники</h5>
      <div className="sources-container">
        {version.sources.map((source, index) => (
          <div key={index} className="source-card">
            <div className="source-header">
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
              {version.sources.length > 1 && (
                <button
                  type="button"
                  className="source-remove-btn"
                  onClick={() => handleSourceRemove(index)}
                  disabled={loading}
                >
                  <i className="fas fa-trash-alt"></i>
                </button>
              )}
            </div>
            <input
              type="url"
              placeholder="https://..."
              value={source.url ?? ''}
              onChange={(e) => handleSourceChange(index, 'url', e.target.value)}
              className="source-input"
              disabled={loading}
            />
            <input
              type="text"
              placeholder="Описание источника"
              value={source.description}
              onChange={(e) => handleSourceChange(index, 'description', e.target.value)}
              className="source-input"
              disabled={loading}
            />
          </div>
        ))}
        <button type="button" className="add-source-btn" onClick={handleSourceAdd} disabled={loading}>
          <i className="fas fa-plus-circle"></i> Добавить источник
        </button>
      </div>

      <h5>Файл версии</h5>
      <FileUploader
        onFileSelect={(file) => onVersionChange({ ...version, file })}
        accept=".zip,.tar,.gz,.rar,.7z"
        currentFile={version.file}
      />
      <div className="form-actions">
        <button className="button small" onClick={onSubmit} disabled={loading}>
          {loading ? 'Сохранение...' : 'Сохранить'}
        </button>
        <button className="button small secondary" onClick={onCancel} disabled={loading}>
          Отмена
        </button>
      </div>
    </div>
  );
};

export default AddVersionForm;
