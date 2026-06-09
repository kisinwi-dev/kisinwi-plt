import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { MLModel } from '../../types/mlModels';
import { formatDateTime } from '../../utils/format';

interface Props {
  model: MLModel;
}

const ModelCard: React.FC<Props> = ({ model }) => {
  const navigate = useNavigate();
  const open = () => navigate(`/models/${model.id}`);

  return (
    <div
      className="card card--hoverable model-card"
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
      <div className="model-card-header">
        <div className="model-title-group">
          <h2>{model.name}</h2>
          <span className="model-version" title="Версия">
            <i className="fas fa-code-branch"></i> v{model.version}
          </span>
        </div>
        <span className={`status-badge status-${model.status}`}>{model.status}</span>
      </div>

      <div className="model-meta">
        {model.model_type && (
          <span title="Тип модели">
            <i className="fas fa-cube"></i>
            <span className="meta-label">Тип:</span> {model.model_type}
          </span>
        )}
        {model.framework && (
          <span title="Фреймворк">
            <i className="fas fa-layer-group"></i>
            <span className="meta-label">Фреймворк:</span> {model.framework}
            {model.framework_version ? ` ${model.framework_version}` : ''}
          </span>
        )}
        <span title="Количество классов">
          <i className="fas fa-tags"></i>
          <span className="meta-label">Классов:</span> {model.classes.length}
        </span>
        <span title="Дата создания">
          <i className="fas fa-calendar-alt"></i>
          <span className="meta-label">Создана:</span> {formatDateTime(model.created_at)}
        </span>
      </div>

      {model.description && (
        <p className="model-description">{model.description}</p>
      )}
    </div>
  );
};

export default ModelCard;
