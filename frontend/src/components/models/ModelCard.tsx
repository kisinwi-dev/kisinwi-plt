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
      className="model-card"
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
          <span><i className="fas fa-cube"></i> {model.model_type}</span>
        )}
        {model.framework && (
          <span>
            <i className="fas fa-layer-group"></i> {model.framework}
            {model.framework_version ? ` ${model.framework_version}` : ''}
          </span>
        )}
        <span><i className="fas fa-tags"></i> {model.classes.length} классов</span>
        <span><i className="fas fa-calendar-alt"></i> {formatDateTime(model.created_at)}</span>
      </div>

      {model.description && (
        <p className="model-description">{model.description}</p>
      )}
    </div>
  );
};

export default ModelCard;
