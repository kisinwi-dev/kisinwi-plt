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
import { metricsService } from '../../services/metricsService';
import { usePolling } from '../../hooks';
import { POLL_INTERVAL_METRICS_MS } from '../../constants';
import { ICONS } from '../../constants/icons';

interface Props {
  modelId: string;
}

const CHART_COLOR = '#b15e6b';
const GRID_COLOR = 'rgba(255,255,255,0.06)';
const AXIS_COLOR = '#b0b8c5';

const toChartData = (values: number[]) =>
  values.map((v, i) => ({ epoch: i + 1, value: v }));

const formatValue = (v: number) =>
  Number.isInteger(v) ? String(v) : v.toFixed(4);

const ModelMetricsCharts: React.FC<Props> = ({ modelId }) => {
  const { data, loading, error } = usePolling(
    () => metricsService.getModelMetrics(modelId),
    { intervalMs: POLL_INTERVAL_METRICS_MS, deps: [modelId] },
  );

  const metrics = data?.metrics ?? [];
  // Ошибку показываем только если так и не получили данные (первая загрузка упала).
  const status: 'loading' | 'empty' | 'error' | 'ready' =
    loading && data === undefined ? 'loading'
    : error && data === undefined ? 'error'
    : metrics.length === 0 ? 'empty'
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
        <i className={`fas ${ICONS.metrics}`}></i> Данные по эпохам отсутствуют.
      </div>
    );
  }

  return (
    <div className="metrics-charts">
      {metrics.map((metric) => (
        <div key={metric.name} className="metrics-chart-block">
          <p className="metrics-chart-title">{metric.name}</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={toChartData(metric.values)} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
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
                  border: '1px solid var(--color-border-strong)',
                  borderRadius: 8,
                  fontSize: 12,
                  color: 'var(--color-text-primary)',
                }}
                formatter={(value) => [formatValue(Number(value)), metric.name]}
                labelFormatter={(label) => `Эпоха ${label}`}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke={CHART_COLOR}
                strokeWidth={2}
                dot={metric.values.length <= 40}
                activeDot={{ r: 4, fill: CHART_COLOR }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
};

export default ModelMetricsCharts;
