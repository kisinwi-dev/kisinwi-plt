import React, { useState } from 'react';
import { formatBytes, formatDateTime } from '../../utils/format';
import type { Dataset, VersionSplitsResponse } from '../../types/dataset';
import { datasetService } from '../../services/datasetService';
import VersionSplitsStats from './VersionSplitsStats';
import { useNotification } from '../../contexts/NotificationContext';

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
  const [versionStats, setVersionStats] = useState<Record<string, VersionSplitsResponse | 'loading' | 'error'>>({});
  const { showNotification } = useNotification();

  const handleCopyId = () => {
    navigator.clipboard.writeText(dataset.id);
    showNotification('ID скопирован', 'success');
  };

  const handleShowVersionStats = async (versionId: string) => {
    if (versionStats[versionId]) {
      setVersionStats(prev => { const next = { ...prev }; delete next[versionId]; return next; });
      return;
    }
    setVersionStats(prev => ({ ...prev, [versionId]: 'loading' }));
    try {
      const data = await datasetService.getVersionSplits(dataset.id, versionId);
      setVersionStats(prev => ({ ...prev, [versionId]: data }));
    } catch {
      setVersionStats(prev => ({ ...prev, [versionId]: 'error' }));
    }
  };

  return (
    <div className="card dataset-card">
      <div className="dataset-header">
        <div className="dataset-title-group">
          <h2>{dataset.name}</h2>
          <span
            className="dataset-id"
            title="Нажмите, чтобы скопировать ID"
            onClick={handleCopyId}
          >
            <i className="fas fa-hashtag"></i>{dataset.id}
            <i className="fas fa-copy dataset-id-copy-icon"></i>
          </span>
        </div>
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
        <span><i className="fas fa-calendar-alt"></i> Создан: {formatDateTime(dataset.created_at)}</span>
        <span><i className="fas fa-sync-alt"></i> Обновлён: {formatDateTime(dataset.updated_at)}</span>
      </div>

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
                      onClick={() => handleShowVersionStats(ver.id)}
                      title="Статистика"
                    >
                      <i className="fas fa-chart-bar"></i>
                    </button>
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
                {versionStats[ver.id] === 'loading' && (
                  <p className="stats-loading">Загрузка статистики...</p>
                )}
                {versionStats[ver.id] === 'error' && (
                  <p className="stats-error">Не удалось загрузить статистику</p>
                )}
                {typeof versionStats[ver.id] === 'object' && (
                  <VersionSplitsStats
                    stats={versionStats[ver.id] as VersionSplitsResponse}
                    onClose={() => handleShowVersionStats(ver.id)}
                  />
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
