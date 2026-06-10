import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { MLModel, MLModelVersion } from '../../types/mlModels';
import { formatDateParts, formatDateTime } from '../../utils/format';
import { mlModelsService } from '../../services/mlModelsService';
import { useNotification } from '../../contexts/NotificationContext';
import ConfirmModal from '../common/ConfirmModal';
import { Tooltip } from '../common/Tooltip';
import { ICONS } from '../../constants/icons';

interface Props {
  model: MLModel;
  onReload: () => void;
}

type PendingDelete =
  | { kind: 'version'; version: MLModelVersion }
  | { kind: 'model'; id: string; name: string; count: number };

const ModelGroupCard: React.FC<Props> = ({ model, onReload }) => {
  const navigate = useNavigate();
  const { showNotification } = useNotification();
  const latest = model.versions[0];
  const [sortAsc, setSortAsc] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [pending, setPending] = useState<PendingDelete | null>(null);

  const sorted = [...model.versions].sort((a, b) =>
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
        await mlModelsService.deleteVersion(pending.version.id);
        showNotification(`Версия v${pending.version.version} удалена`, 'success');
      } else {
        await mlModelsService.deleteModel(pending.id);
        showNotification(`Модель «${pending.name}» удалена со всеми версиями`, 'success');
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
              <h2>{model.name}</h2>
              <span className="model-group-count">
                <i className={`fas ${ICONS.version}`}></i> {model.versions.length} {model.versions.length === 1 ? 'версия' : model.versions.length < 5 ? 'версии' : 'версий'}
              </span>
            </div>
            <Tooltip content="Удалить модель со всеми версиями">
              <button
                className="icon-button icon-button--danger"
                aria-label="Удалить модель со всеми версиями"
                onClick={(e) => { e.stopPropagation(); setPending({ kind: 'model', id: model.id, name: model.name, count: model.versions.length }); }}
              >
                <i className={`fas ${ICONS.delete}`}></i>
              </button>
            </Tooltip>
          </div>
          <div className="model-meta model-group-meta">
            {latest?.framework && (
              <Tooltip content="Фреймворк (последняя версия)">
                <span>
                  <i className={`fas ${ICONS.framework}`}></i>
                  <span className="meta-label">Фреймворк:</span> {latest.framework}
                  {latest.framework_version ? ` ${latest.framework_version}` : ''}
                </span>
              </Tooltip>
            )}
            {latest && (
              <Tooltip content="Количество классов (последняя версия)">
                <span>
                  <i className={`fas ${ICONS.classes}`}></i>
                  <span className="meta-label">Классов:</span> {latest.classes.length}
                </span>
              </Tooltip>
            )}
            <Tooltip content="Дата последней версии">
              <span>
                <i className={`fas ${ICONS.dateUpdated}`}></i>
                <span className="meta-label">Обновлена:</span> {formatDateTime(latest ? latest.created_at : model.created_at)}
              </span>
            </Tooltip>
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
                >
                  <Tooltip content={sortAsc ? 'Сейчас: старые → новые. Нажать для обратного порядка' : 'Сейчас: новые → старые. Нажать для обратного порядка'}>
                    Версия
                    <span className={`sort-icon${sortAsc ? ' sort-asc' : ' sort-desc'}`}>
                      <i className={`fas ${ICONS.collapse} sort-icon-up`}></i>
                      <i className={`fas ${ICONS.expand} sort-icon-down`}></i>
                    </span>
                  </Tooltip>
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
                    <Tooltip content="Удалить версию">
                      <button
                        className="icon-button icon-button--danger small"
                        aria-label="Удалить версию"
                        onClick={(e) => { e.stopPropagation(); setPending({ kind: 'version', version: v }); }}
                      >
                        <i className={`fas ${ICONS.delete}`}></i>
                      </button>
                    </Tooltip>
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
        title={pending?.kind === 'model' ? 'Удалить модель?' : 'Удалить версию?'}
        message={
          pending?.kind === 'model'
            ? `Модель «${pending.name}» и все её ${pending.count} ${pending.count === 1 ? 'версия' : pending.count < 5 ? 'версии' : 'версий'} будут удалены безвозвратно.`
            : pending?.kind === 'version'
            ? `Версия v${pending.version.version} модели «${model.name}» будет удалена безвозвратно.`
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
