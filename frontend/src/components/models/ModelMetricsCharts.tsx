import React, { useMemo, useState } from 'react';
import type { ModelMetrics, Split } from '../../services/metricsService';
import type { UseModelMetricsStreamResult } from '../../hooks';
import { ICONS } from '../../constants/icons';
import { getMetricMeta, getMetricQuality, SPLIT_DESCRIPTIONS } from '../../constants/metrics';
import { formatMetricValue } from '../../utils/format';
import { Tooltip as AppTooltip, InfoHint } from '../common/Tooltip';
import { CollapseChevron, getDisclosureProps } from '../common/Collapse';
import ModelClassReport from './ModelClassReport';
import { CurveChartGrid, CurveZoomOverlay } from './curveChart';
import { yDomain } from './curveChartData';
import type { CheckpointMark, CurveGridItem, CurveSeries, EpochChart } from './curveChartData';

interface Props {
  modelId: string;
  /** Текстовый отчёт об обучении (model.metrics_report) — рендерится в блоке test. */
  metricsReport?: string | null;
  /** SSE-стрим метрик: живёт в ModelDetail, чтобы статус обучения был и в шапке. */
  stream: UseModelMetricsStreamResult;
}

const SPLIT_COLORS: Record<Split, string> = {
  train: '#b15e6b',
  val: '#5e8ab1',
  test: '#8a93a3',
};

// Пунктир эпохи сохранённых весов: нейтральный серый, чтобы не спорить
// с цветами выборок train/val.
const CKPT_COLOR = '#8a93a3';

// Кривые по эпохам: train и val накладываются на один график метрики —
// так виден разрыв между ними (основной сигнал переобучения), как в W&B/TensorBoard.
const CURVE_SPLITS: Split[] = ['train', 'val'];

interface MetricChart extends EpochChart {
  splits: Split[];
}

interface ScalarCard {
  name: string;
  split: Split;
  value: number;
}

// Серии train/val из >1 точки — оверлей-графики; одиночные значения train/val
// и весь test (финальная оценка — одна точка) — скалярные карточки.
// test-карточки — отдельный массив: они уходят в блок «Итоговая оценка (test)».
const splitMetricViews = (
  data: ModelMetrics,
): { charts: MetricChart[]; testScalars: ScalarCard[]; trainValScalars: ScalarCard[] } => {
  const trainValScalars: ScalarCard[] = [];
  const byName = new Map<string, Partial<Record<Split, number[]>>>();

  for (const split of CURVE_SPLITS) {
    for (const metric of data[split]) {
      if (metric.values.length === 0) continue;
      if (metric.values.length === 1) {
        trainValScalars.push({ name: metric.name, split, value: metric.values[0] });
        continue;
      }
      const series = byName.get(metric.name) ?? {};
      series[split] = metric.values;
      byName.set(metric.name, series);
    }
  }

  const testScalars: ScalarCard[] = [];
  for (const metric of data.test) {
    if (metric.values.length === 0) continue;
    testScalars.push({ name: metric.name, split: 'test', value: metric.values[metric.values.length - 1] });
  }

  const splitOrder: Record<Split, number> = { test: 0, val: 1, train: 2 };
  trainValScalars.sort((a, b) => splitOrder[a.split] - splitOrder[b.split]);

  const charts = Array.from(byName.entries()).map(([name, series]) => {
    const splits = CURVE_SPLITS.filter((split) => series[split] !== undefined);
    const maxPoints = Math.max(...splits.map((split) => series[split]!.length));
    const rows = Array.from({ length: maxPoints }, (_, i) => {
      const row: EpochChart['rows'][number] = { epoch: i + 1 };
      for (const split of splits) {
        const value = series[split]![i];
        if (value !== undefined) row[split] = value;
      }
      return row;
    });
    return { name, splits, rows, maxPoints, domain: yDomain(rows, splits) };
  });

  return { charts, testScalars, trainValScalars };
};

// Карточка финального значения метрики; общая для блоков test и train/val.
// В блоке test заголовок секции уже называет выборку, поэтому бейдж split скрыт.
const ScalarCardView: React.FC<ScalarCard & { showSplit?: boolean }> = ({ name, split, value, showSplit = true }) => {
  const meta = getMetricMeta(name);
  const quality = getMetricQuality(meta, value);
  return (
    <div
      className={`metrics-scalar-card${quality ? ` metrics-scalar-card--${quality}` : ''}`}
      style={{ '--split-color': SPLIT_COLORS[split] } as React.CSSProperties}
    >
      <span className="metrics-scalar-card-head">
        <AppTooltip content={meta?.description}>
          <span className="metrics-scalar-card-name">
            {meta && <i className={`fas ${meta.icon}`} aria-hidden="true"></i>} {name}
          </span>
        </AppTooltip>
        {showSplit && (
          <AppTooltip content={SPLIT_DESCRIPTIONS[split]}>
            <span className="metrics-scalar-card-split">{split}</span>
          </AppTooltip>
        )}
      </span>
      <span className="metrics-scalar-card-value">{formatMetricValue(value)}</span>
    </div>
  );
};

const ModelMetricsCharts: React.FC<Props> = ({ modelId, metricsReport, stream }) => {
  const { data, finished, loading, error } = stream;
  const [reportOpen, setReportOpen] = useState(false);

  // Сборка графиков — самая дорогая часть страницы (recharts): мемоизируем,
  // чтобы не пересчитывать при рендерах, не меняющих данные метрик.
  const { charts, testScalars, trainValScalars } = useMemo(
    () => (data ? splitMetricViews(data) : { charts: [], testScalars: [], trainValScalars: [] }),
    [data],
  );

  // «Сохранённые веса»: эпоха чекпоинта по early-stop-метрике. Общее число
  // эпох выводим из длины кривых train/val; у старых моделей чекпоинта нет.
  const checkpoint = data?.checkpoint;
  const totalEpochs = data
    ? Math.max(0, ...[...data.train, ...data.val].map((m) => m.values.length))
    : 0;

  // Серии и чекпоинты графиков для общей сетки/оверлея. Чекпоинт у модели
  // один (нейтральный серый пунктир); в сводке оверлея он показывается
  // в карточке val (early-stop считается на валидации), иначе — в первой.
  const gridItems: CurveGridItem[] = useMemo(
    () =>
      charts.map((chart) => {
        const ckptSplit = chart.splits.includes('val') ? 'val' : chart.splits[0];
        const series: CurveSeries[] = chart.splits.map((split) => ({
          key: split,
          label: split,
          color: SPLIT_COLORS[split],
          hint: SPLIT_DESCRIPTIONS[split],
          checkpointEpoch: split === ckptSplit ? checkpoint?.epoch : undefined,
        }));
        const checkpoints: CheckpointMark[] =
          checkpoint && checkpoint.epoch <= chart.maxPoints
            ? [{ epoch: checkpoint.epoch, color: CKPT_COLOR }]
            : [];
        return { chart, series, checkpoints };
      }),
    [charts, checkpoint],
  );

  // Увеличенный график в оверлее: имя метрики или null.
  const [zoomedName, setZoomedName] = useState<string | null>(null);
  const zoomedItem = gridItems.find((item) => item.chart.name === zoomedName) ?? null;

  // Ошибку показываем только если так и не получили данные (первая загрузка упала).
  const status: 'loading' | 'empty' | 'error' | 'ready' =
    loading ? 'loading'
    : error && data === undefined ? 'error'
    : charts.length === 0 && testScalars.length === 0 && trainValScalars.length === 0 ? 'empty'
    : 'ready';

  // Текстовый отчёт об обучении: часть итоговой оценки, но показываем и когда
  // числовых метрик нет (старые модели) — после плейсхолдера.
  const textReport = metricsReport ? (
    <div className="metrics-report-details">
      <div
        className="metrics-report-summary"
        {...getDisclosureProps(reportOpen, () => setReportOpen((o) => !o))}
      >
        <CollapseChevron open={reportOpen} />
        <i className={`fas ${ICONS.report}`}></i> Текстовый отчёт
        <InfoHint text="Полный отчёт в текстовом виде, сформированный при обучении модели." />
      </div>
      {reportOpen && <pre className="detail-pre metrics-report-pre">{metricsReport}</pre>}
    </div>
  ) : null;

  if (status === 'loading') {
    return (
      <div className="metrics-charts-placeholder">
        <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка графиков…
      </div>
    );
  }

  if (status === 'error') {
    return (
      <>
        <div className="metrics-charts-placeholder metrics-charts-placeholder--error">
          <i className={`fas ${ICONS.warning}`}></i> Не удалось загрузить данные из сервиса метрик.
        </div>
        {textReport}
      </>
    );
  }

  if (status === 'empty') {
    return (
      <>
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.metrics}`}></i> Данные по эпохам отсутствуют.
        </div>
        {textReport}
      </>
    );
  }

  return (
    <>
      {/* finished: отчёт по классам возможен только после завершения обучения. */}
      {(testScalars.length > 0 || textReport || finished) && (
        <div className="metrics-split-section">
          <p className="metrics-split-title">
            Итоговая оценка (test)
            <InfoHint text="Финальное качество модели на test-выборке — данных, которые она не видела при обучении. Это главный показатель того, как модель поведёт себя на новых данных." />
          </p>
          {checkpoint && (
            <p className="metrics-checkpoint-note">
              <i className={`fas ${ICONS.weights}`} aria-hidden="true"></i>
              Сохранены веса эпохи {checkpoint.epoch}
              {totalEpochs ? ` из ${totalEpochs}` : ''} — лучшая по val_{checkpoint.metric}
              {checkpoint.value != null ? ` (${formatMetricValue(checkpoint.value)})` : ''}
              <InfoHint text="Trainer сохраняет веса лучшей эпохи по early-stop-метрике и восстанавливает их перед тестированием — test-метрики соответствуют именно этим весам. На графиках эта эпоха отмечена пунктирной вертикалью." />
            </p>
          )}
          {testScalars.length > 0 && (
            <div className="metrics-scalar-cards">
              {testScalars.map((card) => (
                <ScalarCardView key={`${card.split}-${card.name}`} {...card} showSplit={false} />
              ))}
            </div>
          )}
          <ModelClassReport modelId={modelId} enabled={finished} />
          {textReport}
        </div>
      )}
      {(charts.length > 0 || trainValScalars.length > 0) && (
        <div className="metrics-split-section">
          <p className="metrics-split-title">
            Динамика обучения (train / val)
            <InfoHint text="Как метрики менялись по эпохам на обучающей (train) и валидационной (val) выборках. Если val заметно хуже train — модель переобучилась: заучила примеры вместо закономерностей. Пунктирная вертикаль — эпоха сохранённых весов; кнопкой справа график разворачивается с приближением диапазона эпох." />
          </p>
          {/* Порядок графиков пользовательский — общий для всех моделей. */}
          <CurveChartGrid
            items={gridItems}
            storageKey="model-metrics-curves-order"
            onZoom={setZoomedName}
          />
          {trainValScalars.length > 0 && (
            <div className="metrics-scalar-cards">
              {trainValScalars.map((card) => (
                <ScalarCardView key={`${card.split}-${card.name}`} {...card} />
              ))}
            </div>
          )}
        </div>
      )}
      {zoomedItem && (
        <CurveZoomOverlay
          title={zoomedItem.chart.name}
          chart={zoomedItem.chart}
          series={zoomedItem.series}
          checkpoints={zoomedItem.checkpoints}
          onClose={() => setZoomedName(null)}
        />
      )}
    </>
  );
};

export default ModelMetricsCharts;
