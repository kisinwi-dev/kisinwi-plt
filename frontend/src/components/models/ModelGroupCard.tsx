import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { MLModelGroup, MLModel } from '../../types/mlModels';
import { formatDateParts, formatDateTime } from '../../utils/format';
import { mlModelsService } from '../../services/mlModelsService';
import { useNotification } from '../../contexts/NotificationContext';
import ConfirmModal from '../common/ConfirmModal';
import { ICONS } from '../../constants/icons';

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
  const [expanded, setExpanded] = useState(false);
  const [pending, setPending] = useState<PendingDelete | null>(null);

  const sorted = [...group.versions].sort((a, b) =>
    sortAsc ? Number(a.version) - Number(b.version) : Number(b.version) - Number(a.version)
  );

  // По умолчанию показываем только COLLAPSED_COUNT свежих версий, чтобы не растягивать
  // страницу. Остальные — под кнопкой «показать все».
  const COLLAPSED_COUNT = 3;
  const collapsible = sorted.length > COLLAPSED_COUNT;
  const visible = collapsible && !expanded ? sorted.slice(0, COLLAPSED_COUNT) : sorted;
  const hiddenCount = sorted.length - visible.length;

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
          <div className="model-group-header-top">
            <div className="model-title-group">
              <h2>{group.name}</h2>
              <span className="model-group-count">
                <i className={`fas ${ICONS.version}`}></i> {group.versions.length} {group.versions.length === 1 ? 'версия' : group.versions.length < 5 ? 'версии' : 'версий'}
              </span>
            </div>
            <button
              className="icon-button icon-button--danger"
              title="Удалить все версии"
              onClick={(e) => { e.stopPropagation(); setPending({ kind: 'group', name: group.name, count: group.versions.length }); }}
            >
              <i className={`fas ${ICONS.delete}`}></i>
            </button>
          </div>
          <div className="model-meta model-group-meta">
            {latest.framework && (
              <span title="Фреймворк (последняя версия)">
                <i className={`fas ${ICONS.framework}`}></i>
                <span className="meta-label">Фреймворк:</span> {latest.framework}
                {latest.framework_version ? ` ${latest.framework_version}` : ''}
              </span>
            )}
            <span title="Количество классов (последняя версия)">
              <i className={`fas ${ICONS.classes}`}></i>
              <span className="meta-label">Классов:</span> {latest.classes.length}
            </span>
            <span title="Дата последней версии">
              <i className={`fas ${ICONS.dateUpdated}`}></i>
              <span className="meta-label">Обновлена:</span> {formatDateTime(latest.created_at)}
            </span>
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
                    <i className={`fas ${ICONS.collapse} sort-icon-up`}></i>
                    <i className={`fas ${ICONS.expand} sort-icon-down`}></i>
                  </span>
                </th>
                <th>Тип</th>
                <th>Статус</th>
                <th>Дата</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {visible.map((v) => (
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
                    <i className={`fas ${ICONS.version}`}></i> v{v.version}
                  </td>
                  <td className="model-version-type">{v.model_type ?? '—'}</td>
                  <td><span className={`status-badge status-${v.status}`}>{v.status}</span></td>
                  <td className="model-version-date">
                    {(() => { const { date, time } = formatDateParts(v.created_at); return <><span>{date}</span>{time && <span className="model-version-time">{time}</span>}</>; })()}
                  </td>
                  <td className="model-version-actions">
                    <button
                      className="icon-button icon-button--danger small"
                      title="Удалить версию"
                      onClick={(e) => { e.stopPropagation(); setPending({ kind: 'version', model: v }); }}
                    >
                      <i className={`fas ${ICONS.delete}`}></i>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {collapsible && (
            <button
              className="versions-toggle"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={expanded}
            >
              {expanded
                ? <><i className={`fas ${ICONS.collapse}`}></i> Свернуть</>
                : <><i className={`fas ${ICONS.expand}`}></i> Показать ещё {hiddenCount}</>}
            </button>
          )}
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
