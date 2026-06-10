import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import { formatDateParts } from '../utils/format';
import type { Dataset } from '../types/dataset';
import type { VersionComparisonResponse } from '../types/datasetComparison';
import Select from '../components/common/Select';
import { Tooltip } from '../components/common/Tooltip';
import CompareSummaryCards from '../components/datasets/compare/CompareSummaryCards';
import CompareCountsSection from '../components/datasets/compare/CompareCountsSection';
import CompareDistributionSection from '../components/datasets/compare/CompareDistributionSection';
import CompareBalanceSection from '../components/datasets/compare/CompareBalanceSection';
import CompareSizeStatsSection from '../components/datasets/compare/CompareSizeStatsSection';
import CompareFilesSection from '../components/datasets/compare/CompareFilesSection';
import { ICONS } from '../constants/icons';
import './Datasets.css';
import './DatasetCompare.css';

/**
 * Страница сравнения двух версий датасета.
 * Выбранные версии живут в query string (?from=&to=), поэтому ссылкой
 * на конкретное сравнение можно делиться.
 */
const DatasetCompare: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { showNotification } = useNotification();
  const [searchParams, setSearchParams] = useSearchParams();

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(true);
  // Результат сравнения помечен ключом «from|to»: результат для устаревшей
  // пары версий игнорируется, а статус загрузки — производное значение.
  const [result, setResult] = useState<
    { key: string; status: 'ready'; data: VersionComparisonResponse } | { key: string; status: 'error' } | null
  >(null);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    datasetService.getDataset(id)
      .then(data => {
        if (!cancelled) setDataset(data);
      })
      .catch(err => {
        if (!cancelled) {
          showNotification(err instanceof Error ? err.message : 'Не удалось загрузить датасет', 'error');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [id, showNotification]);

  // Версии — от новых к старым, как в списке на странице датасета.
  const versions = useMemo(() => {
    const list = [...(dataset?.versions ?? [])];
    list.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    return list;
  }, [dataset?.versions]);

  const fromParam = searchParams.get('from');
  const toParam = searchParams.get('to');

  // Дефолты и валидация query-параметров: невалидные id (удалённая версия,
  // опечатка в ссылке) сбрасываются; пустые заполняются — «сравниваемая» =
  // самая новая версия, «базовая» = следующая за ней.
  useEffect(() => {
    if (versions.length < 2) return;
    const ids = new Set(versions.map(v => v.id));
    let from = fromParam && ids.has(fromParam) ? fromParam : null;
    let to = toParam && ids.has(toParam) ? toParam : null;
    if ((fromParam && !from) || (toParam && !to)) {
      showNotification('Версия из ссылки не найдена', 'warning');
    }
    if (!to) to = versions.find(v => v.id !== from)?.id ?? null;
    if (!from) from = versions.find(v => v.id !== to)?.id ?? null;
    if (from && to && (from !== fromParam || to !== toParam)) {
      setSearchParams({ from, to }, { replace: true });
    }
  }, [versions, fromParam, toParam, setSearchParams, showNotification]);

  const versionById = useMemo(() => new Map(versions.map(v => [v.id, v])), [versions]);
  const fromVersion = fromParam ? versionById.get(fromParam) : undefined;
  const toVersion = toParam ? versionById.get(toParam) : undefined;
  const fromId = fromVersion?.id;
  const toId = toVersion?.id;

  const compareKey = id && fromId && toId && fromId !== toId ? `${fromId}|${toId}` : null;

  useEffect(() => {
    if (!id || !fromId || !toId || fromId === toId) return;
    const key = `${fromId}|${toId}`;
    let cancelled = false;
    datasetService.compareVersions(id, fromId, toId)
      .then(data => {
        if (!cancelled) setResult({ key, status: 'ready', data });
      })
      .catch(() => {
        if (!cancelled) setResult({ key, status: 'error' });
      });
    return () => { cancelled = true; };
  }, [id, fromId, toId, retryKey]);

  const currentResult = result && result.key === compareKey ? result : null;
  const comparison = currentResult?.status === 'ready' ? currentResult.data : null;
  const compareLoading = compareKey !== null && currentResult === null;
  const compareError = currentResult?.status === 'error';

  const versionOptions = useMemo(
    () =>
      versions.map(v => ({
        value: v.id,
        label: `${v.name} · ${formatDateParts(v.created_at).date}`,
      })),
    [versions],
  );

  const setVersionParam = (key: 'from' | 'to', value: string) => {
    const next = new URLSearchParams(searchParams);
    next.set(key, value);
    setSearchParams(next, { replace: true });
  };

  const swapVersions = () => {
    if (!fromParam || !toParam) return;
    setSearchParams({ from: toParam, to: fromParam }, { replace: true });
  };

  if (loading) {
    return (
      <div className="page">
        <div className="loading-state">
          <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка датасета…
        </div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="page">
        <button className="button secondary" onClick={() => navigate('/datasets')}>
          <i className={`fas ${ICONS.back}`}></i> К списку
        </button>
        <div className="empty-state">
          <i className={`fas ${ICONS.notFound}`}></i> Датасет не найден.
        </div>
      </div>
    );
  }

  return (
    <div className="page dataset-detail">
      <button className="detail-back-link" onClick={() => navigate(`/datasets/${dataset.id}`)}>
        <i className={`fas ${ICONS.back}`}></i> К датасету
      </button>

      <header className="dataset-detail-header">
        <div className="dataset-detail-heading">
          <div className="dataset-detail-title">
            <h1>Сравнение версий</h1>
            <span className="dataset-badge">
              <i className={`fas ${ICONS.dataset}`}></i> {dataset.name}
            </span>
          </div>
        </div>
      </header>

      {versions.length < 2 ? (
        <div className="empty-state">
          <i className={`fas ${ICONS.empty}`}></i> Для сравнения нужно минимум две версии.
        </div>
      ) : (
        <>
          <section className="detail-section vcmp-picker">
            <div className="vcmp-picker-field">
              <span className="vcmp-picker-label">
                <i className={`fas ${ICONS.version}`}></i> Базовая версия
              </span>
              <Select
                value={fromParam ?? ''}
                options={versionOptions}
                onChange={(v) => setVersionParam('from', v)}
                ariaLabel="Базовая версия"
              />
            </div>
            <Tooltip content="Поменять версии местами">
              <button
                className="icon-button vcmp-swap"
                onClick={swapVersions}
                aria-label="Поменять версии местами"
              >
                <i className={`fas ${ICONS.swap}`}></i>
              </button>
            </Tooltip>
            <div className="vcmp-picker-field">
              <span className="vcmp-picker-label">
                <i className={`fas ${ICONS.version}`}></i> Сравниваемая версия
              </span>
              <Select
                value={toParam ?? ''}
                options={versionOptions}
                onChange={(v) => setVersionParam('to', v)}
                ariaLabel="Сравниваемая версия"
              />
            </div>
          </section>

          {fromId && toId && fromId === toId && (
            <div className="vstats-placeholder">
              <i className={`fas ${ICONS.info}`}></i> Выберите две разные версии.
            </div>
          )}

          {compareLoading && (
            <div className="vstats-placeholder">
              <i className={`fas ${ICONS.loading} fa-spin`}></i> Сравнение версий…
            </div>
          )}

          {compareError && (
            <div className="vstats-placeholder vstats-placeholder--error">
              <i className={`fas ${ICONS.warning}`}></i> Не удалось сравнить версии.
              <button
                className="button secondary small"
                onClick={() => { setResult(null); setRetryKey(k => k + 1); }}
              >
                Повторить
              </button>
            </div>
          )}

          {comparison && id && fromVersion && toVersion && (
            <>
              <CompareSummaryCards
                comparison={comparison}
                fromName={fromVersion.name}
                toName={toVersion.name}
              />
              <CompareCountsSection
                counts={comparison.counts}
                fromName={fromVersion.name}
                toName={toVersion.name}
              />
              <CompareDistributionSection distribution={comparison.distribution} />
              <CompareBalanceSection balance={comparison.balance} />
              <CompareSizeStatsSection
                sizeStats={comparison.size_stats}
                fromName={fromVersion.name}
                toName={toVersion.name}
              />
              <CompareFilesSection
                key={`${fromVersion.id}_${toVersion.id}`}
                datasetId={id}
                fromVersionId={fromVersion.id}
                toVersionId={toVersion.id}
                summary={comparison.files}
              />
            </>
          )}
        </>
      )}
    </div>
  );
};

export default DatasetCompare;
