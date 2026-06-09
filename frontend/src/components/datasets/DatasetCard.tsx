import React from 'react';
import { useNavigate } from 'react-router-dom';
import { formatDateTime } from '../../utils/format';
import { useCopyToClipboard } from '../../hooks';
import { ICONS } from '../../constants/icons';
import type { Dataset } from '../../types/dataset';

interface DatasetCardProps {
  dataset: Dataset;
  onDelete: (dataset: Dataset) => void;
}

// Сколько классов показываем в превью карточки, остальные сворачиваем в "+N ещё"
const CLASSES_PREVIEW_LIMIT = 10;

// Компактное превью датасета. Клик открывает страницу с полной информацией.
const DatasetCard: React.FC<DatasetCardProps> = ({ dataset, onDelete }) => {
  const navigate = useNavigate();
  const open = () => navigate(`/datasets/${dataset.id}`);
  const copyToClipboard = useCopyToClipboard();

  const handleCopyId = (e: React.MouseEvent) => {
    e.stopPropagation();
    copyToClipboard(dataset.id);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete(dataset);
  };

  return (
    <div
      className="card card--hoverable dataset-card"
      onClick={open}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          open();
        }
      }}
    >
      <div className="dataset-header">
        <div className="dataset-title-group">
          <span className="dataset-badge"><i className={`fas ${ICONS.tag}`}></i> {dataset.type} / {dataset.task}</span>
          <h2>{dataset.name}</h2>
          <span
            className="dataset-id"
            title="Нажмите, чтобы скопировать ID"
            onClick={handleCopyId}
          >
            <i className={`fas ${ICONS.id}`}></i>{dataset.id}
            <i className={`fas ${ICONS.copy} dataset-id-copy-icon`}></i>
          </span>
        </div>
        <div className="dataset-actions">
          <button
            className="icon-button icon-button--danger"
            title="Удалить датасет"
            aria-label="Удалить датасет"
            onClick={handleDelete}
          >
            <i className={`fas ${ICONS.delete}`}></i>
          </button>
        </div>
      </div>

      <div className="dataset-meta">
        <span><i className={`fas ${ICONS.classes}`}></i> Классов: {dataset.classes_count}</span>
        <span><i className={`fas ${ICONS.version}`}></i> Версий: {dataset.versions.length}</span>
      </div>

      <div className="dataset-meta dataset-meta--dates">
        <span title="Создан"><i className={`fas ${ICONS.dateCreated}`}></i> Создан: {formatDateTime(dataset.created_at)}</span>
        <span title="Изменён"><i className={`fas ${ICONS.dateUpdated}`}></i> Изменён: {formatDateTime(dataset.updated_at)}</span>
      </div>

      {dataset.classes_names.length > 0 && (
        <div className="dataset-classes">
          <span className="dataset-classes-label">Классы:</span>
          <div className="tag-list">
            {dataset.classes_names.slice(0, CLASSES_PREVIEW_LIMIT).map(className => (
              <span key={className} className="tag">{className}</span>
            ))}
            {dataset.classes_names.length > CLASSES_PREVIEW_LIMIT && (
              <span className="tag tag--more">
                +{dataset.classes_names.length - CLASSES_PREVIEW_LIMIT} ещё
              </span>
            )}
          </div>
        </div>
      )}

      {dataset.description && (
        <p className="dataset-description dataset-description--clamped">
          <span className="dataset-description-label">Описание: </span>
          {dataset.description}
        </p>
      )}
    </div>
  );
};

export default DatasetCard;
