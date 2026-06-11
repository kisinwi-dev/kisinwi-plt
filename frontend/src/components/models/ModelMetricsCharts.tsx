import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { ModelMetrics, Split, TrainingStatus } from '../../services/metricsService';
import { useModelMetricsStream } from '../../hooks';
import { ICONS } from '../../constants/icons';
import ModelClassReport from './ModelClassReport';

interface Props {
  modelId: string;
}

const SPLIT_COLORS: Record<Split, string> = {
  train: '#b15e6b',
  val: '#5e8ab1',
  test: '#8a93a3',
};
const GRID_COLOR = 'rgba(255,255,255,0.06)';
const AXIS_COLOR = '#b0b8c5';

// Кривые по эпохам: train и val накладываются на один график метрики —
// так виден разрыв между ними (основной сигнал переобучения), как в W&B/TensorBoard.
const CURVE_SPLITS: Split[] = ['train', 'val'];

interface MetricChart {
  name: string;
  splits: Split[];
  rows: Array<{ epoch: number } & Partial<Record<Split, number>>>;
  maxPoints: number;
}

interface ScalarCard {
  name: string;
  split: Split;
  value: number;
}

// Серии train/val из >1 точки — оверлей-графики; одиночные значения любой выборки
// и весь test (финальная оценка — одна точка) — скалярные карточки.
const splitMetricViews = (data: ModelMetrics): { charts: MetricChart[]; scalars: ScalarCard[] } => {
  const scalars: ScalarCard[] = [];
  const byName = new Map<string, Partial<Record<Split, number[]>>>();

  for (const split of CURVE_SPLITS) {
    for (const metric of data[split]) {
      if (metric.values.length === 0) continue;
      if (metric.values.length === 1) {
        scalars.push({ name: metric.name, split, value: metric.values[0] });
        continue;
      }
      const series = byName.get(metric.name) ?? {};
      series[split] = metric.values;
      byName.set(metric.name, series);
    }
  }

  for (const metric of data.test) {
    if (metric.values.length === 0) continue;
    scalars.push({ name: metric.name, split: 'test', value: metric.values[metric.values.length - 1] });
  }

  const charts = Array.from(byName.entries()).map(([name, series]) => {
    const splits = CURVE_SPLITS.filter((split) => series[split] !== undefined);
    const maxPoints = Math.max(...splits.map((split) => series[split]!.length));
    const rows = Array.from({ length: maxPoints }, (_, i) => {
      const row: MetricChart['rows'][number] = { epoch: i + 1 };
      for (const split of splits) {
        const value = series[split]![i];
        if (value !== undefined) row[split] = value;
      }
      return row;
    });
    return { name, splits, rows, maxPoints };
  });

  return { charts, scalars };
};

const formatValue = (v: number) =>
  Number.isInteger(v) ? String(v) : v.toFixed(4);

// Бейдж статуса обучения; у старых моделей статуса нет — бейдж не показываем.
const STATUS_BADGES: Record<TrainingStatus, { icon: string; spin?: boolean; label: string }> = {
  in_progress: { icon: ICONS.loading, spin: true, label: 'Обучение идёт…' },
  completed: { icon: ICONS.success, label: 'Обучение завершено' },
  failed: { icon: ICONS.error, label: 'Ошибка обучения' },
  cancelled: { icon: ICONS.cancelled, label: 'Обучение отменено' },
};

const TrainingStatusBadge: React.FC<{ status: TrainingStatus }> = ({ status }) => {
  const badge = STATUS_BADGES[status];
  return (
    <span className={`metrics-status-badge metrics-status-badge--${status}`}>
      <i className={`fas ${badge.icon}${badge.spin ? ' fa-spin' : ''}`}></i> {badge.label}
    </span>
  );
};

const ModelMetricsCharts: React.FC<Props> = ({ modelId }) => {
  const { data, status: trainingStatus, finished, loading, error } = useModelMetricsStream(modelId);

  const { charts, scalars } = data ? splitMetricViews(data) : { charts: [], scalars: [] };
  // Ошибку показываем только если так и не получили данные (первая загрузка упала).
  const status: 'loading' | 'empty' | 'error' | 'ready' =
    loading ? 'loading'
    : error && data === undefined ? 'error'
    : charts.length === 0 && scalars.length === 0 ? 'empty'
    : 'ready';

  if (status === 'loading') {
    return (
      <div className="metrics-charts-placeholder">
        <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка графиков…
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="metrics-charts-placeholder metrics-charts-placeholder--error">
        <i className={`fas ${ICONS.warning}`}></i> Не удалось загрузить данные из сервиса метрик.
      </div>
    );
  }

  if (status === 'empty') {
    return (
      <div className="metrics-charts-placeholder">
        {trainingStatus && <TrainingStatusBadge status={trainingStatus} />}
        <i className={`fas ${ICONS.metrics}`}></i> Данные по эпохам отсутствуют.
      </div>
    );
  }

  return (
    <>
      {trainingStatus && (
        <div className="metrics-status-row">
          <TrainingStatusBadge status={trainingStatus} />
        </div>
      )}
      {scalars.length > 0 && (
        <div className="metrics-split-section">
          <p className="metrics-split-title">Финальные значения</p>
          <div className="metrics-scalar-cards">
            {scalars.map(({ name, split, value }) => (
              <div key={`${split}-${name}`} className="metrics-scalar-card">
                <span className="metrics-scalar-card-name">
                  <span style={{ color: SPLIT_COLORS[split] }}>●</span> {split} · {name}
                </span>
                <span className="metrics-scalar-card-value">{formatValue(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {charts.length > 0 && (
        <div className="metrics-split-section">
          <p className="metrics-split-title">Кривые по эпохам</p>
          <div className="metrics-charts">
            {charts.map((chart) => (
              <div key={chart.name} className="metrics-chart-block">
                <p className="metrics-chart-title">
                  {chart.name}
                  <span style={{ marginLeft: 12, fontSize: 11 }}>
                    {chart.splits.map((split) => (
                      <span key={split} style={{ color: SPLIT_COLORS[split], marginRight: 8 }}>
                        ● {split}
                      </span>
                    ))}
                  </span>
                </p>
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={chart.rows} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} />
                    <XAxis
                      dataKey="epoch"
                      tick={{ fill: AXIS_COLOR, fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: GRID_COLOR }}
                      label={{ value: 'эпоха', position: 'insideBottomRight', offset: -4, fill: AXIS_COLOR, fontSize: 10 }}
                    />
                    <YAxis
                      tick={{ fill: AXIS_COLOR, fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={formatValue}
                      width={56}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'var(--color-bg-secondary)',
                        border: '1px solid var(--color-border-soft)',
                        borderRadius: 12,
                        boxShadow: 'var(--shadow-md)',
                        fontSize: 12,
                        color: 'var(--color-text-primary)',
                      }}
                      formatter={(value, name) => [formatValue(Number(value)), String(name)]}
                      labelFormatter={(label) => `Эпоха ${label}`}
                    />
                    {chart.splits.map((split) => (
                      <Line
                        key={split}
                        type="monotone"
                        dataKey={split}
                        stroke={SPLIT_COLORS[split]}
                        strokeWidth={2}
                        dot={chart.maxPoints <= 40}
                        activeDot={{ r: 4, fill: SPLIT_COLORS[split] }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        </div>
      )}
      <ModelClassReport modelId={modelId} enabled={finished} />
    </>
  );
};

export default ModelMetricsCharts;
