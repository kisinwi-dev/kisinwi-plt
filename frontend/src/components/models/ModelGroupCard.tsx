import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { MLModelGroup, MLModel } from '../../types/mlModels';
import { formatDateParts } from '../../utils/format';
import { mlModelsService } from '../../services/mlModelsService';
import { useNotification } from '../../contexts/NotificationContext';
import ConfirmModal from '../common/ConfirmModal';

interface Props {
  group: MLModelGroup;
  onReload: () => void;
}

type PendingDelete =
  | { kind: 'version'; model: MLModel }
  | { kind: 'group'; name: string; count: number };

const ModelGroupCard: React.FC<Props> = ({ group, onReload }) => {
  const navigate = useNavigate();
  const { showNotification } = useNotification();
  const latest = group.versions[0];
  const [sortAsc, setSortAsc] = useState(false);
  const [pending, setPending] = useState<PendingDelete | null>(null);

  const sorted = [...group.versions].sort((a, b) =>
    sortAsc ? Number(a.version) - Number(b.version) : Number(b.version) - Number(a.version)
  );

  const handleConfirmDelete = async () => {
    if (!pending) return;
    setPending(null);
    try {
      if (pending.kind === 'version') {
        await mlModelsService.deleteModel(pending.model.id);
        showNotification(`Версия v${pending.model.version} удалена`, 'success');
      } else {
        await mlModelsService.deleteModelsByName(pending.name);
        showNotification(`Все версии модели «${pending.name}» удалены`, 'success');
      }
      onReload();
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления', 'error');
    }
  };

  return (
    <>
      <div className="card model-group-card">
        <div className="model-group-header">
          <div className="model-title-group">
            <h2>{group.name}</h2>
            <span className="model-group-count">
              <i className="fas fa-code-branch"></i> {group.versions.length} {group.versions.length === 1 ? 'версия' : group.versions.length < 5 ? 'версии' : 'версий'}
            </span>
          </div>
          <div className="model-group-header-actions">
            <div className="model-meta">
              {latest.framework && (
                <span>
                  <i className="fas fa-layer-group"></i> {latest.framework}
                  {latest.framework_version ? ` ${latest.framework_version}` : ''}
                </span>
              )}
              <span><i className="fas fa-tags"></i> {latest.classes.length} классов</span>
            </div>
            <button
              className="btn-icon btn-icon--danger"
              title="Удалить все версии"
              onClick={(e) => { e.stopPropagation(); setPending({ kind: 'group', name: group.name, count: group.versions.length }); }}
            >
              <i className="fas fa-trash"></i>
            </button>
          </div>
        </div>

        <div className="model-group-versions">
          <table className="versions-table">
            <colgroup>
              <col /><col /><col /><col /><col style={{ width: '2.5rem' }} />
            </colgroup>
            <thead>
              <tr>
                <th
                  className="versions-th-sortable"
                  onClick={() => setSortAsc((v) => !v)}
                  aria-sort={sortAsc ? 'ascending' : 'descending'}
                  title={sortAsc ? 'Сейчас: старые → новые. Нажать для обратного порядка' : 'Сейчас: новые → старые. Нажать для обратного порядка'}
                >
                  Версия
                  <span className={`sort-icon${sortAsc ? ' sort-asc' : ' sort-desc'}`}>
                    <i className="fas fa-chevron-up sort-icon-up"></i>
                    <i className="fas fa-chevron-down sort-icon-down"></i>
                  </span>
                </th>
                <th>Тип</th>
                <th>Статус</th>
                <th>Дата</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((v) => (
                <tr
                  key={v.id}
                  className="model-version-row"
                  onClick={() => navigate(`/models/${v.id}`)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      navigate(`/models/${v.id}`);
                    }
                  }}
                >
                  <td className="model-version-label">
                    <i className="fas fa-code-branch"></i> v{v.version}
                  </td>
                  <td className="model-version-type">{v.model_type ?? '—'}</td>
                  <td><span className={`status-badge status-${v.status}`}>{v.status}</span></td>
                  <td className="model-version-date">
                    {(() => { const { date, time } = formatDateParts(v.created_at); return <><span>{date}</span>{time && <span className="model-version-time">{time}</span>}</>; })()}
                  </td>
                  <td className="model-version-actions">
                    <button
                      className="btn-icon btn-icon--danger btn-icon--sm"
                      title="Удалить версию"
                      onClick={(e) => { e.stopPropagation(); setPending({ kind: 'version', model: v }); }}
                    >
                      <i className="fas fa-trash"></i>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <ConfirmModal
        open={pending !== null}
        danger
        title={pending?.kind === 'group' ? 'Удалить все версии?' : 'Удалить версию?'}
        message={
          pending?.kind === 'group'
            ? `Все ${pending.count} ${pending.count === 1 ? 'версия' : pending.count < 5 ? 'версии' : 'версий'} модели «${pending.name}» будут удалены безвозвратно.`
            : pending?.kind === 'version'
            ? `Версия v${pending.model.version} модели «${group.name}» будет удалена безвозвратно.`
            : undefined
        }
        confirmLabel="Удалить"
        cancelLabel="Отмена"
        onConfirm={handleConfirmDelete}
        onCancel={() => setPending(null)}
      />
    </>
  );
};

export default ModelGroupCard;
