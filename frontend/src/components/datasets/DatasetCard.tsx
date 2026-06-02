import React from 'react';
import { formatBytes } from '../../utils/format';
import type { Dataset } from '../../types/dataset';

interface DatasetCardProps {
  dataset: Dataset;
  loading: boolean;
  showVersionForm: boolean;
  onAddVersion: () => void;
  onDelete: () => void;
  onDeleteVersion: (versionId: string) => void;
  versionForm?: React.ReactNode;
}

const DatasetCard: React.FC<DatasetCardProps> = ({
  dataset,
  loading,
  showVersionForm,
  onAddVersion,
  onDelete,
  onDeleteVersion,
  versionForm,
}) => {
  return (
    <div className="dataset-card">
      <div className="dataset-header">
        <h2>{dataset.name}</h2>
        <div className="dataset-actions">
          <button
            className="icon-button"
            onClick={onAddVersion}
            title="Добавить версию"
            disabled={loading}
          >
            <i className="fas fa-code-branch"></i>
          </button>
          <button
            className="icon-button"
            onClick={onDelete}
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

      {dataset.classes_names.length > 0 && (
        <div className="dataset-classes">
          <h4>Классы ({dataset.classes_count})</h4>
          <div className="class-tags">
            {dataset.classes_names.slice(0, 10).map(className => (
              <span key={className} className="class-tag">{className}</span>
            ))}
            {dataset.classes_names.length > 10 && <span className="class-tag">...</span>}
          </div>
        </div>
      )}

      {showVersionForm && versionForm}

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
                      onClick={() => onDeleteVersion(ver.id)}
                      title="Удалить версию"
                      disabled={loading}
                    >
                      <i className="fas fa-trash"></i>
                    </button>
                  </div>
                </div>
                <span className="version-date">Дата загрузки: {new Date(ver.created_at).toLocaleDateString()}</span>
                <p className="version-description">Описание: {ver.description}</p>
                <div className="version-stats">
                  <span>Всего: {ver.num_samples.toLocaleString()}</span>
                </div>
                {ver.sources.length > 0 && (
                  <div className="dataset-sources">
                    {ver.sources.map((src, idx) => (
                      <div key={idx} className="source-item">
                        <span className="source-type-badge">{src.type}</span>
                        {src.url && (
                          <a href={src.url} target="_blank" rel="noopener noreferrer" className="source-link">
                            {src.url}
                          </a>
                        )}
                        {src.description && <span className="source-description">{src.description}</span>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetCard;
