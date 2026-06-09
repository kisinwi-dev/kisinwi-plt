import React from 'react';
import FileUploader from '../FileUploader';
import SourcesEditor from './SourcesEditor';
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

  const handleSourcesChange = (sources: SourceItem[]) => {
    onVersionChange({ ...version, sources });
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
      <SourcesEditor
        sources={version.sources}
        loading={loading}
        onChange={handleSourcesChange}
      />

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
