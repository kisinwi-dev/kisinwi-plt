import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { mlModelsService } from '../services/mlModelsService';
import { metricsService } from '../services/metricsService';
import type { ModelsCompareResponse, Split } from '../services/metricsService';
import { useNotification } from '../contexts/NotificationContext';
import { formatDateParts } from '../utils/format';
import type { MLModelVersion } from '../types/mlModels';
import Select from '../components/common/Select';
import { Tooltip } from '../components/common/Tooltip';
import { ICONS } from '../constants/icons';
import './Models.css';
// Стили блока выбора пары (vcmp-*) общие со сравнением версий датасета.
import './DatasetCompare.css';

const SPLITS: Split[] = ['train', 'val', 'test'];
// Лимит выборки версий для пикеров; пагинацию здесь не делаем.
const VERSIONS_LIMIT = 100;

const formatValue = (v: number) =>
  Number.isInteger(v) ? String(v) : v.toFixed(4);

/**
 * Страница сравнения двух версий моделей по метрикам обучения
 * (POST /models/compare сервиса metrics). Выбранные версии живут в query
 * string (?from=&to=), поэтому ссылкой на сравнение можно делиться.
 */
const ModelCompare: React.FC = () => {
  const navigate = useNavigate();
  const { showNotification } = useNotification();
  const [searchParams, setSearchParams] = useSearchParams();

  const [versions, setVersions] = useState<MLModelVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [split, setSplit] = useState<Split>('val');
  // Результат помечен ключом «from|to|split»: ответ для устаревшего выбора игнорируется.
  const [result, setResult] = useState<
    { key: string; status: 'ready'; data: ModelsCompareResponse } | { key: string; status: 'error' } | null
  >(null);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    mlModelsService.getVersions({ limit: VERSIONS_LIMIT })
      .then((data) => {
        if (!cancelled) setVersions(data.versions);
      })
      .catch((err) => {
        if (!cancelled) {
          showNotification(err instanceof Error ? err.message : 'Не удалось загрузить модели', 'error');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [showNotification]);

  const fromParam = searchParams.get('from');
  const toParam = searchParams.get('to');

  // Дефолты и валидация query-параметров: невалидные id сбрасываются,
  // пустые заполняются самыми свежими версиями (как в сравнении датасетов).
  useEffect(() => {
    if (versions.length < 2) return;
    const ids = new Set(versions.map((v) => v.id));
    let from = fromParam && ids.has(fromParam) ? fromParam : null;
    let to = toParam && ids.has(toParam) ? toParam : null;
    if ((fromParam && !from) || (toParam && !to)) {
      showNotification('Модель из ссылки не найдена', 'warning');
    }
    if (!to) to = versions.find((v) => v.id !== from)?.id ?? null;
    if (!from) from = versions.find((v) => v.id !== to)?.id ?? null;
    if (from && to && (from !== fromParam || to !== toParam)) {
      setSearchParams({ from, to }, { replace: true });
    }
  }, [versions, fromParam, toParam, setSearchParams, showNotification]);

  const versionById = useMemo(() => new Map(versions.map((v) => [v.id, v])), [versions]);
  const fromVersion = fromParam ? versionById.get(fromParam) : undefined;
  const toVersion = toParam ? versionById.get(toParam) : undefined;
  const fromId = fromVersion?.id;
  const toId = toVersion?.id;

  const compareKey = fromId && toId && fromId !== toId ? `${fromId}|${toId}|${split}` : null;

  useEffect(() => {
    if (!fromId || !toId || fromId === toId) return;
    const key = `${fromId}|${toId}|${split}`;
    let cancelled = false;
    metricsService.compareModels([fromId, toId], split)
      .then((data) => {
        if (!cancelled) setResult({ key, status: 'ready', data });
      })
      .catch(() => {
        if (!cancelled) setResult({ key, status: 'error' });
      });
    return () => { cancelled = true; };
  }, [fromId, toId, split, retryKey]);

  const currentResult = result && result.key === compareKey ? result : null;
  const comparison = currentResult?.status === 'ready' ? currentResult.data : null;
  const compareLoading = compareKey !== null && currentResult === null;
  const compareError = currentResult?.status === 'error';

  const versionOptions = useMemo(
    () =>
      versions.map((v) => ({
        value: v.id,
        label: `${v.name} · v${v.version} · ${formatDateParts(v.created_at).date}`,
      })),
    [versions],
  );

  const modelLabel = (modelId: string) => {
    const version = versionById.get(modelId);
    return version ? `${version.name} · v${version.version}` : modelId;
  };

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
          <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка моделей…
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <button className="detail-back-link" onClick={() => navigate('/models')}>
        <i className={`fas ${ICONS.back}`}></i> К моделям
      </button>

      <div className="page-header">
        <h1>Сравнение моделей</h1>
        <p className="page-description">
          Сравнение версий моделей по метрикам обучения: финальные и лучшие значения, отставание от лидера.
        </p>
      </div>

      {versions.length < 2 ? (
        <div className="empty-state">
          <i className={`fas ${ICONS.empty}`}></i> Для сравнения нужно минимум две версии моделей.
        </div>
      ) : (
        <>
          <section className="detail-section vcmp-picker">
            <div className="vcmp-picker-field">
              <span className="vcmp-picker-label">
                <i className={`fas ${ICONS.model}`}></i> Базовая модель
              </span>
              <Select
                value={fromParam ?? ''}
                options={versionOptions}
                onChange={(v) => setVersionParam('from', v)}
                ariaLabel="Базовая модель"
              />
            </div>
            <Tooltip content="Поменять модели местами">
              <button
                className="icon-button vcmp-swap"
                onClick={swapVersions}
                aria-label="Поменять модели местами"
              >
                <i className={`fas ${ICONS.swap}`}></i>
              </button>
            </Tooltip>
            <div className="vcmp-picker-field">
              <span className="vcmp-picker-label">
                <i className={`fas ${ICONS.model}`}></i> Сравниваемая модель
              </span>
              <Select
                value={toParam ?? ''}
                options={versionOptions}
                onChange={(v) => setVersionParam('to', v)}
                ariaLabel="Сравниваемая модель"
              />
            </div>
            <div className="view-toggle mcmp-split-toggle" role="group" aria-label="Выборка для сравнения">
              {SPLITS.map((s) => (
                <button
                  key={s}
                  className={`view-toggle-btn${split === s ? ' active' : ''}`}
                  onClick={() => setSplit(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </section>

          {fromId && toId && fromId === toId && (
            <div className="metrics-charts-placeholder">
              <i className={`fas ${ICONS.info}`}></i> Выберите две разные модели.
            </div>
          )}

          {compareLoading && (
            <div className="metrics-charts-placeholder">
              <i className={`fas ${ICONS.loading} fa-spin`}></i> Сравнение моделей…
            </div>
          )}

          {compareError && (
            <div className="metrics-charts-placeholder metrics-charts-placeholder--error">
              <i className={`fas ${ICONS.warning}`}></i> Не удалось сравнить модели.
              <button
                className="button secondary small"
                onClick={() => { setResult(null); setRetryKey((k) => k + 1); }}
              >
                Повторить
              </button>
            </div>
          )}

          {comparison && (
            <>
              {comparison.missing.length > 0 && (
                <div className="metrics-charts-placeholder">
                  <i className={`fas ${ICONS.info}`}></i>
                  Нет сохранённых метрик: {comparison.missing.map(modelLabel).join(', ')}.
                </div>
              )}

              {comparison.metrics.length === 0 && comparison.missing.length === 0 && (
                <div className="empty-state">
                  <i className={`fas ${ICONS.empty}`}></i> На выборке {comparison.split} нет общих метрик для сравнения.
                </div>
              )}

              {comparison.metrics.map((metric) => (
                <section key={metric.metric} className="detail-section mcmp-metric-block">
                  <p className="metrics-chart-title">
                    {metric.metric}
                    <Tooltip content={metric.higher_is_better ? 'Больше — лучше' : 'Меньше — лучше'}>
                      <span className="mcmp-direction">
                        {metric.higher_is_better ? '↑' : '↓'}
                      </span>
                    </Tooltip>
                  </p>
                  <div className="class-report-table-wrap">
                    <table className="class-report-table mcmp-table">
                      <thead>
                        <tr>
                          <th>Модель</th>
                          <th>Финал</th>
                          <th>Лучшее</th>
                          <th>Лучшая эпоха</th>
                          <th>Эпох</th>
                          <th>Δ от лидера</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metric.models.map((entry) => {
                          const isBest = entry.model_id === metric.best_model_id;
                          return (
                            <tr key={entry.model_id} className={isBest ? 'mcmp-row--best' : undefined}>
                              <th>
                                {isBest && <i className={`fas ${ICONS.star} mcmp-best-icon`}></i>}
                                {modelLabel(entry.model_id)}
                              </th>
                              <td>{formatValue(entry.final_value)}</td>
                              <td>{formatValue(entry.best_value)}</td>
                              <td>{entry.best_epoch}</td>
                              <td>{entry.epochs}</td>
                              <td>{isBest ? '—' : formatValue(entry.delta_best)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>
              ))}
            </>
          )}
        </>
      )}
    </div>
  );
};

export default ModelCompare;
