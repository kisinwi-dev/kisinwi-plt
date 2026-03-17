import React from 'react';
import FileUploader from '../FileUploader';
import type { NewVersion } from '../../types/dataset';

interface AddVersionFormProps {
  version: NewVersion & { file: File | null };
  loading: boolean;
  onVersionChange: (version: NewVersion & { file: File | null }) => void;
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
  const handleChange = (field: keyof NewVersion, value: string) => {
    onVersionChange({ ...version, [field]: value });
  };

  return (
    <div className="add-version-form">
      <h5>Добавить версию</h5>
      <input
        type="text"
        placeholder="ID версии *"
        value={version.version_id}
        onChange={(e) => handleChange('version_id', e.target.value)}
        disabled={loading}
      />
      <input
        type="text"
        placeholder="Описание"
        value={version.description}
        onChange={(e) => handleChange('description', e.target.value)}
        disabled={loading}
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