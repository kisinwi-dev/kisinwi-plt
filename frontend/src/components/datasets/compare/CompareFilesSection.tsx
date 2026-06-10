import React, { useEffect, useMemo, useState } from 'react';
import { ICONS } from '../../../constants/icons';
import { CollapseChevron, getDisclosureProps } from '../../common/Collapse';
import { Tooltip } from '../../common/Tooltip';
import { datasetService } from '../../../services/datasetService';
import type { FilesDiffResponse, FilesDiffSummary } from '../../../types/datasetComparison';

interface Props {
  datasetId: string;
  fromVersionId: string;
  toVersionId: string;
  /** Счётчики из общей сводки — доступны до загрузки списков. */
  summary: FilesDiffSummary;
}

// Порция рендера длинных списков файлов.
const FILES_PAGE = 200;

interface FileListProps {
  title: string;
  kind: 'added' | 'removed';
  paths: string[];
}

const FileList: React.FC<FileListProps> = ({ title, kind, paths }) => {
  const [shown, setShown] = useState(FILES_PAGE);
  if (paths.length === 0) return null;
  const sign = kind === 'added' ? '+' : '−';
  return (
    <div className="vcmp-file-list-block">
      <h5 className="vstats-subtitle">
        <i className={`fas ${kind === 'added' ? ICONS.add : ICONS.minus}`}></i>
        {title} ({paths.length.toLocaleString()})
      </h5>
      <div className="vcmp-file-list">
        {paths.slice(0, shown).map(path => (
          <div key={path} className={`vcmp-file-row vcmp-file-row--${kind}`}>
            {sign} {path}
          </div>
        ))}
      </div>
      {paths.length > shown && (
        <button
          className="button secondary small vcmp-show-more"
          onClick={() => setShown(n => n + FILES_PAGE)}
        >
          Показать ещё {Math.min(FILES_PAGE, paths.length - shown).toLocaleString()} (осталось{' '}
          {(paths.length - shown).toLocaleString()})
        </button>
      )}
    </div>
  );
};

/**
 * По-файловый diff в git-стиле: списки добавленных/удалённых файлов.
 * Списки могут быть большими, поэтому грузятся лениво — при первом раскрытии.
 */
const CompareFilesSection: React.FC<Props> = ({ datasetId, fromVersionId, toVersionId, summary }) => {
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState<FilesDiffResponse | null>(null);
  // Тело секции рендерится только раскрытым, поэтому стартовое состояние — загрузка.
  const [status, setStatus] = useState<'loading' | 'error' | 'ready'>('loading');
  const [retryKey, setRetryKey] = useState(0);
  const [filter, setFilter] = useState('');

  const hasChanges = summary.added_count > 0 || summary.removed_count > 0;

  useEffect(() => {
    if (!open || files !== null) return;
    let cancelled = false;
    datasetService.compareVersionFiles(datasetId, fromVersionId, toVersionId)
      .then(data => {
        if (cancelled) return;
        setFiles(data);
        setStatus('ready');
      })
      .catch(() => {
        if (!cancelled) setStatus('error');
      });
    return () => { cancelled = true; };
  }, [open, files, retryKey, datasetId, fromVersionId, toVersionId]);

  const [added, removed] = useMemo(() => {
    if (!files) return [[], []] as [string[], string[]];
    const q = filter.trim().toLowerCase();
    if (!q) return [files.added, files.removed];
    return [
      files.added.filter(p => p.toLowerCase().includes(q)),
      files.removed.filter(p => p.toLowerCase().includes(q)),
    ];
  }, [files, filter]);

  return (
    <section className="detail-section vcmp-section">
      <h3
        className={`detail-section-title${hasChanges ? ' vcmp-disclosure' : ''}`}
        {...(hasChanges ? getDisclosureProps(open, () => setOpen(o => !o)) : {})}
      >
        {hasChanges && <CollapseChevron open={open} />}
        <i className={`fas ${ICONS.file}`}></i> Файлы
        <Tooltip content="Пофайловый diff: какие файлы добавлены и какие удалены относительно базовой версии.">
          <i
            className={`fas ${ICONS.info} vcmp-title-info`}
            onClick={e => e.stopPropagation()}
          ></i>
        </Tooltip>
        <span className="vcmp-files-counts">
          <span className="vcmp-delta--up">+{summary.added_count.toLocaleString()}</span>
          {' / '}
          <span className="vcmp-delta--down">−{summary.removed_count.toLocaleString()}</span>
        </span>
        <span className="vcmp-card-note">{summary.common_count.toLocaleString()} без изменений</span>
      </h3>

      {!hasChanges && <p className="vstats-empty">Файловых изменений нет.</p>}

      {hasChanges && open && (
        <>
          {status === 'loading' && (
            <div className="vstats-placeholder">
              <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка списка файлов…
            </div>
          )}
          {status === 'error' && (
            <div className="vstats-placeholder vstats-placeholder--error">
              <i className={`fas ${ICONS.warning}`}></i> Не удалось загрузить список файлов.
              <button
                className="button secondary small"
                onClick={() => { setStatus('loading'); setRetryKey(k => k + 1); }}
              >
                Повторить
              </button>
            </div>
          )}
          {status === 'ready' && files && (
            <>
              <div className="filter-field versions-filter">
                <i className={`fas ${ICONS.search}`}></i>
                <input
                  type="text"
                  placeholder="Фильтр по пути файла"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                />
                {filter && (
                  <Tooltip content="Сбросить фильтр">
                    <button
                      type="button"
                      className="versions-filter-clear"
                      onClick={() => setFilter('')}
                      aria-label="Сбросить фильтр"
                    >
                      <i className={`fas ${ICONS.close}`}></i>
                    </button>
                  </Tooltip>
                )}
              </div>
              {added.length === 0 && removed.length === 0 ? (
                <p className="vstats-empty">Ничего не найдено по фильтру.</p>
              ) : (
                <>
                  <FileList title="Добавленные" kind="added" paths={added} />
                  <FileList title="Удалённые" kind="removed" paths={removed} />
                </>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
};

export default CompareFilesSection;
