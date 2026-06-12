import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { mlModelsService } from '../services/mlModelsService';
import { metricsService } from '../services/metricsService';
import type { ModelMetrics, ModelsCompareResponse } from '../services/metricsService';
import { useNotification } from '../contexts/NotificationContext';
import { formatDateParts } from '../utils/format';
import type { MLModelVersion } from '../types/mlModels';
import Select from '../components/common/Select';
import { Tooltip, InfoHint } from '../components/common/Tooltip';
import { ICONS } from '../constants/icons';
import ModelCompareSummary from '../components/models/ModelCompareSummary';
import ModelCompareTable from '../components/models/ModelCompareTable';
import ModelCompareCurves from '../components/models/ModelCompareCurves';
import ModelCompareMetaDiff from '../components/models/ModelCompareMetaDiff';
import ModelCompareClassReport from '../components/models/ModelCompareClassReport';
import { COMPARE_COLORS, MAX_COMPARE_MODELS, totalEpochsOf } from '../components/models/modelCompare';
import type { CompareSide, WeightsInfo } from '../components/models/modelCompare';
import './Models.css';
// Стили блока выбора (vcmp-*) общие со сравнением версий датасета.
import './DatasetCompare.css';
import './ModelCompare.css';

// Лимит выборки версий для пикеров; пагинацию здесь не делаем.
const VERSIONS_LIMIT = 100;

/**
 * Страница сравнения версий моделей (от 2 до MAX_COMPARE_MODELS): вердикт,
 * сводная diff-таблица метрик (POST /models/compare сервиса metrics),
 * наложенные кривые обучения, diff конфигураций и сравнение по классам.
 * Сравнение всегда идёт по test: тестовые метрики измерены на итоговых
 * сохранённых весах каждой модели — весах эпохи с лучшим значением loss
 * на валидационной выборке (или иной early-stop-метрики из конфига),
 * поэтому именно они честно отвечают «какая модель лучше». Кривые train/val
 * показывают сам процесс обучения. Выбранные версии живут в query string
 * (?ids=a,b,c — первая базовая), поэтому ссылкой на сравнение можно делиться;
 * старый формат ?from=&to= поддерживается для обратной совместимости.
 */
const ModelCompare: React.FC = () => {
  const navigate = useNavigate();
  const { showNotification } = useNotification();
  const [searchParams, setSearchParams] = useSearchParams();

  const [versions, setVersions] = useState<MLModelVersion[]>([]);
  const [loading, setLoading] = useState(true);
  // Результат помечен набором моделей: ответ для устаревшего выбора игнорируется.
  const [result, setResult] = useState<
    | { idsKey: string; status: 'ready'; data: ModelsCompareResponse }
    | { idsKey: string; status: 'error' }
    | null
  >(null);
  const [retryKey, setRetryKey] = useState(0);
  // Кэш удачных ответов compareModels по набору моделей: возврат к прежнему
  // выбору показывает результат мгновенно, без запроса.
  const compareCacheRef = useRef(new Map<string, ModelsCompareResponse>());
  // Поэпоховые метрики моделей для наложенных кривых: getModelMetrics отдаёт
  // все сплиты сразу. Ошибка кривых не блокирует таблицу.
  const [curves, setCurves] = useState<
    | { key: string; status: 'ready'; data: Array<ModelMetrics | null> }
    | { key: string; status: 'error' }
    | null
  >(null);
  const [curvesRetryKey, setCurvesRetryKey] = useState(0);

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

  // Выбранные id: ?ids=a,b,c; старые ссылки ?from=&to= конвертируются.
  const idsParam = searchParams.get('ids');
  const legacyParam = [searchParams.get('from'), searchParams.get('to')]
    .filter((v): v is string => Boolean(v))
    .join(',');
  const rawIds = useMemo(
    () => (idsParam ?? legacyParam).split(',').filter(Boolean),
    [idsParam, legacyParam],
  );

  const setIds = (ids: string[]) => {
    setSearchParams({ ids: ids.join(',') }, { replace: true });
  };

  // Дефолты и валидация query-параметров: невалидные id и дубли сбрасываются,
  // выбор дополняется самыми свежими версиями до минимум двух.
  useEffect(() => {
    if (versions.length < 2) return;
    const known = new Set(versions.map((v) => v.id));
    const found = rawIds.filter((id) => known.has(id));
    // Дубли id схлопываем молча — «не найдена» только про реально неизвестные.
    const valid = Array.from(new Set(found)).slice(0, MAX_COMPARE_MODELS);
    if (found.length < rawIds.length) {
      showNotification('Часть моделей из ссылки не найдена', 'warning');
    }
    const filled = [...valid];
    for (const v of versions) {
      if (filled.length >= 2) break;
      if (!filled.includes(v.id)) filled.push(v.id);
    }
    if (filled.join(',') !== rawIds.join(',')) setIds(filled);
    // setIds — стабильная обёртка setSearchParams.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [versions, rawIds, showNotification]);

  const versionById = useMemo(() => new Map(versions.map((v) => [v.id, v])), [versions]);
  const selectedVersions = useMemo(
    () =>
      rawIds
        .map((id) => versionById.get(id))
        .filter((v): v is MLModelVersion => Boolean(v)),
    [rawIds, versionById],
  );
  const selectionValid =
    selectedVersions.length >= 2 && selectedVersions.length === rawIds.length;

  const selectedIds = selectedVersions.map((v) => v.id);
  const idsKey = selectionValid ? selectedIds.join('|') : null;

  useEffect(() => {
    if (!idsKey) return;
    const cached = compareCacheRef.current.get(idsKey);
    if (cached) {
      setResult({ idsKey, status: 'ready', data: cached });
      return;
    }
    let cancelled = false;
    // Сравнение всегда по test: метрики измерены на сохранённых весах.
    metricsService.compareModels(idsKey.split('|'), 'test')
      .then((data) => {
        compareCacheRef.current.set(idsKey, data);
        if (!cancelled) setResult({ idsKey, status: 'ready', data });
      })
      .catch(() => {
        if (!cancelled) setResult({ idsKey, status: 'error' });
      });
    return () => { cancelled = true; };
  }, [idsKey, retryKey]);

  useEffect(() => {
    if (!idsKey) return;
    const ids = idsKey.split('|');
    let cancelled = false;
    // 404 (нет метрик у модели) — это null, а не ошибка.
    Promise.all(ids.map((id) => metricsService.getModelMetrics(id)))
      .then((data) => {
        if (!cancelled) setCurves({ key: idsKey, status: 'ready', data });
      })
      .catch(() => {
        if (!cancelled) setCurves({ key: idsKey, status: 'error' });
      });
    return () => { cancelled = true; };
  }, [idsKey, curvesRetryKey]);

  const currentResult = result && idsKey && result.idsKey === idsKey ? result : null;
  const comparison = currentResult?.status === 'ready' ? currentResult.data : null;
  const compareLoading = idsKey !== null && currentResult === null;
  const compareError = currentResult?.status === 'error';

  const currentCurves = curves && curves.key === idsKey ? curves : null;
  const curvesStatus: 'loading' | 'error' | 'ready' =
    currentCurves === null ? 'loading' : currentCurves.status === 'error' ? 'error' : 'ready';

  // Мемоизация: sides и metricsBySide уходят в мемоизированные секции
  // с recharts-графиками — новые ссылки на каждый рендер заставляли бы
  // перерисовывать все графики при любом изменении состояния страницы.
  const sides: CompareSide[] = useMemo(
    () =>
      selectedVersions.map((v, index) => ({
        id: v.id,
        label: `${v.name} · v${v.version}`,
        name: v.name,
        version: `v${v.version}`,
        color: COMPARE_COLORS[index % COMPARE_COLORS.length],
      })),
    [selectedVersions],
  );
  const emptyMetricsBySide = useMemo(() => sides.map(() => null), [sides]);
  const metricsBySide =
    currentCurves?.status === 'ready' ? currentCurves.data : emptyMetricsBySide;
  const retryCurves = useCallback(() => {
    setCurves(null);
    setCurvesRetryKey((k) => k + 1);
  }, []);

  // Происхождение весов каждой стороны: эпоха чекпоинта, early-stop-метрика
  // и общее число эпох (из длины кривых train/val — на test entry.epochs
  // всегда 1). Уходит в шапку diff-таблицы метрик.
  const weightsBySide: Array<WeightsInfo | null> = useMemo(
    () =>
      metricsBySide.map((metrics) => {
        const ckpt = metrics?.checkpoint;
        if (!ckpt) return null;
        return {
          epoch: ckpt.epoch,
          metric: ckpt.metric,
          value: ckpt.value,
          totalEpochs: totalEpochsOf(metrics),
        };
      }),
    [metricsBySide],
  );

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

  const replaceId = (index: number, value: string) => {
    // Выбор уже участвующей модели в другой позиции — меняем их местами,
    // чтобы не плодить дубли.
    const next = [...selectedIds];
    const existing = next.indexOf(value);
    if (existing >= 0) next[existing] = next[index];
    next[index] = value;
    setIds(next);
  };

  const removeId = (index: number) => {
    if (selectedIds.length <= 2) return;
    setIds(selectedIds.filter((_, i) => i !== index));
  };

  const makeBase = (index: number) => {
    const next = [...selectedIds];
    const [id] = next.splice(index, 1);
    setIds([id, ...next]);
  };

  const addModel = () => {
    const free = versions.find((v) => !selectedIds.includes(v.id));
    if (!free || selectedIds.length >= MAX_COMPARE_MODELS) return;
    setIds([...selectedIds, free.id]);
  };

  const canAdd =
    selectedIds.length < MAX_COMPARE_MODELS && selectedIds.length < versions.length;

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
          Какая модель лучше и почему: вердикт по тестовым метрикам, кривые обучения и различия в конфигурации.
        </p>
      </div>

      {versions.length < 2 ? (
        <div className="empty-state">
          <i className={`fas ${ICONS.empty}`}></i> Для сравнения нужно минимум две версии моделей.
        </div>
      ) : (
        <>
          <section className="detail-section mcmp-picker">
            <div className="mcmp-picker-rows">
              {selectedIds.map((id, index) => (
                <div key={index} className="mcmp-picker-row">
                  <div className="vcmp-picker-field">
                    <span className="vcmp-picker-label">
                      <i className={`fas ${ICONS.model}`}></i>
                      {index === 0 ? 'Базовая модель' : `Модель ${index + 1}`}
                    </span>
                    <Select
                      value={id}
                      options={versionOptions}
                      onChange={(v) => replaceId(index, v)}
                      ariaLabel={index === 0 ? 'Базовая модель' : `Модель ${index + 1}`}
                    />
                  </div>
                  {index > 0 && (
                    <Tooltip content="Сделать базовой: от базовой модели считаются дельты">
                      <button
                        className="icon-button mcmp-row-action"
                        onClick={() => makeBase(index)}
                        aria-label="Сделать базовой"
                      >
                        <i className={`fas ${ICONS.toTop}`}></i>
                      </button>
                    </Tooltip>
                  )}
                  {selectedIds.length > 2 && (
                    <Tooltip content="Убрать из сравнения">
                      <button
                        className="icon-button mcmp-row-action"
                        onClick={() => removeId(index)}
                        aria-label="Убрать из сравнения"
                      >
                        <i className={`fas ${ICONS.close}`}></i>
                      </button>
                    </Tooltip>
                  )}
                </div>
              ))}
              <button
                className="button secondary small mcmp-add-model"
                onClick={addModel}
                disabled={!canAdd}
              >
                <i className={`fas ${ICONS.add}`}></i> Добавить модель
                {selectedIds.length >= MAX_COMPARE_MODELS && ` (макс. ${MAX_COMPARE_MODELS})`}
              </button>
            </div>
          </section>

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

          {comparison && selectionValid && (
            <>
              {comparison.missing.length > 0 && (
                <div className="metrics-charts-placeholder">
                  <i className={`fas ${ICONS.info}`}></i>
                  Нет сохранённых метрик: {comparison.missing.map(modelLabel).join(', ')}.
                </div>
              )}

              {comparison.metrics.length === 0 && comparison.missing.length === 0 && (
                <div className="empty-state">
                  <i className={`fas ${ICONS.empty}`}></i> На тестовой выборке нет общих метрик для сравнения.
                </div>
              )}

              <ModelCompareSummary sides={sides} metrics={comparison.metrics} />

              <section className="detail-section">
                <p className="metrics-split-title">
                  Метрики (test)
                  <InfoHint text="Метрики на тестовой выборке. Сравниваются сохранённые веса каждой модели — веса эпохи с лучшим значением loss на валидационной выборке (если в конфиге не задана другая early-stop-метрика); именно они уходят в реестр и в продакшен. Звезда — лидер по метрике, цвет значения — качество по общепринятым порогам. Под значениями небазовых моделей — дельта от базовой: ▲/▼ — выросло или упало значение, зелёный — модель лучше базовой, красный — хуже (для loss «лучше» — это ▼)." />
                </p>
                {comparison.metrics.length > 0 && (
                  <>
                    <p className="mcmp-section-desc">
                      Тестируются сохранённые веса каждой модели — лучшие по loss
                      на валидационной выборке; в шапке колонки указано, с какой
                      эпохи они взяты. ★ — лидер по метрике; под значениями
                      небазовых моделей — дельта от базовой (▲ выросло,
                      ▼ упало; зелёная — лучше, красная — хуже).
                    </p>
                    <ModelCompareTable
                      sides={sides}
                      metrics={comparison.metrics}
                      weightsBySide={weightsBySide}
                    />
                  </>
                )}
                <ModelCompareClassReport sides={sides} />
              </section>
            </>
          )}

          {selectionValid && (
            <>
              <ModelCompareCurves
                sides={sides}
                split="val"
                metricsBySide={metricsBySide}
                status={curvesStatus}
                onRetry={retryCurves}
                defaultOpen
              />
              <ModelCompareCurves
                sides={sides}
                split="train"
                metricsBySide={metricsBySide}
                status={curvesStatus}
                onRetry={retryCurves}
                defaultOpen={false}
              />
              <ModelCompareMetaDiff sides={sides} versions={selectedVersions} metricsBySide={metricsBySide} />
            </>
          )}
        </>
      )}
    </div>
  );
};

export default ModelCompare;
