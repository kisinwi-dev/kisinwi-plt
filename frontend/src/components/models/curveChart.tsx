import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceDot,
  ResponsiveContainer,
  Brush,
} from 'recharts';
import { ICONS } from '../../constants/icons';
import { getMetricMeta, getMetricQuality } from '../../constants/metrics';
import { formatMetricValue } from '../../utils/format';
import { Tooltip as UiTooltip } from '../common/Tooltip';
import { useDragReorder } from '../../hooks';
import { curveStats } from './curveChartData';
import type { CheckpointMark, CurveGridItem, CurveSeries, EpochChart } from './curveChartData';

/**
 * Общие компоненты графиков train/val-кривых по эпохам: сетка с drag-reorder
 * и кнопкой увеличения, сам график (recharts) и zoom-оверлей со сводкой.
 * Используются страницей модели (серии — выборки train/val) и страницей
 * сравнения (серии — модели); различия параметризованы CurveSeries
 * (типы и расчётные хелперы — в curveChartData.ts).
 */

// Тики оси Y после паддинга домена — дробные; 4 знака перегружают ось,
// поэтому округляем до 3 значащих цифр.
const formatTick = (v: number) => String(Number(v.toPrecision(3)));

/**
 * Легенда графика: цветные маркеры серий с пояснением в тултипе (hint).
 * С onHoverSeries (оверлей) наведение на серию подсвечивает её кривую.
 */
const ChartLegend: React.FC<{
  series: CurveSeries[];
  onHoverSeries?: (key: string | null) => void;
}> = ({ series, onHoverSeries }) => (
  <span className="mcmp-curve-legend">
    {series.map((s) => (
      <UiTooltip key={s.key} content={s.hint}>
        <span
          style={{ color: s.color }}
          onMouseEnter={onHoverSeries && (() => onHoverSeries(s.key))}
          onMouseLeave={onHoverSeries && (() => onHoverSeries(null))}
        >
          ● {s.label}
        </span>
      </UiTooltip>
    ))}
  </span>
);

/**
 * Tooltip увеличенного графика: строки серий отсортированы по значению,
 * у каждой — маркер цвета серии (дефолтный recharts-tooltip их не различает).
 */
const CurveTooltip: React.FC<{
  series: CurveSeries[];
  active?: boolean;
  label?: number;
  payload?: Array<{ dataKey?: string | number; value?: number | string }>;
}> = ({ series, active, label, payload }) => {
  if (!active || !payload?.length) return null;
  const byKey = new Map(series.map((s) => [s.key, s]));
  const rows = payload
    .map((item) => ({ series: byKey.get(String(item.dataKey)), value: Number(item.value) }))
    .filter((row): row is { series: CurveSeries; value: number } =>
      row.series !== undefined && Number.isFinite(row.value),
    )
    .sort((a, b) => b.value - a.value);
  return (
    <div className="mcmp-tooltip">
      <p className="mcmp-tooltip-label">Эпоха {label}</p>
      {rows.map(({ series: s, value }) => (
        <p key={s.key} className="mcmp-tooltip-row">
          <span className="mcmp-side-dot" style={{ background: s.color }} />
          <span className="mcmp-tooltip-name">{s.label}</span>
          <b>{formatMetricValue(value)}</b>
        </p>
      ))}
    </div>
  );
};

/**
 * Отрисовка одного графика кривых; используется и в сетке,
 * и в увеличенном оверлее (там выше порог точек и высота контейнера).
 */
export const CurveChartView: React.FC<{
  chart: EpochChart;
  /** Серии графика — только те, у которых есть значения в rows. */
  series: CurveSeries[];
  /** Пунктирные вертикали эпох сохранённых весов (уже отфильтрованы по эпохам). */
  checkpoints: CheckpointMark[];
  height: number | `${number}%`;
  /** До скольких эпох рисовать точки на линиях. */
  dotLimit: number;
  /** Увеличенный вид: brush для приближения диапазона эпох, крупнее оси и линии. */
  large?: boolean;
  /** Ключ выделенной серии (hover по легенде): остальные кривые приглушаются. */
  highlightKey?: string | null;
}> = ({ chart, series, checkpoints, height, dotLimit, large, highlightKey }) => {
  // Цвета сетки/осей — через CSS-класс (mcmp-chart-themed), чтобы следовали теме.
  const dim = (key: string) => (highlightKey == null || highlightKey === key ? 1 : 0.18);
  const meta = getMetricMeta(chart.name);
  return (
    <ResponsiveContainer width="100%" height={height} className="mcmp-chart-themed">
      <LineChart data={chart.rows} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={!large} />
        {/* Пунктирная вертикаль — эпоха, веса которой сохранены у модели. */}
        {checkpoints.map((ckpt, index) => (
          <ReferenceLine
            key={`ckpt-${ckpt.seriesKey ?? index}`}
            x={ckpt.epoch}
            stroke={ckpt.color}
            strokeDasharray="4 4"
            strokeOpacity={0.7 * (ckpt.seriesKey ? dim(ckpt.seriesKey) : 1)}
          />
        ))}
        <XAxis
          dataKey="epoch"
          tick={{ fontSize: large ? 12 : 11 }}
          tickLine={false}
          // В увеличенном виде подпись налезала бы на полосу brush — там ось
          // поясняется подсказкой под графиком.
          label={
            large
              ? undefined
              : { value: 'эпоха', position: 'insideBottomRight', offset: -4, fontSize: 10 }
          }
        />
        <YAxis
          domain={chart.domain}
          tick={{ fontSize: large ? 12 : 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={formatTick}
          width={56}
        />
        {large ? (
          <Tooltip content={<CurveTooltip series={series} />} />
        ) : (
          <Tooltip
            contentStyle={{
              background: 'var(--color-bg-secondary)',
              border: '1px solid var(--color-border-soft)',
              borderRadius: 12,
              boxShadow: 'var(--shadow-md)',
              fontSize: 12,
              color: 'var(--color-text-primary)',
            }}
            formatter={(value, name) => [
              formatMetricValue(Number(value)),
              series.find((s) => s.key === String(name))?.label ?? String(name),
            ]}
            labelFormatter={(label) => `Эпоха ${label}`}
          />
        )}
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            stroke={s.color}
            strokeWidth={large ? 2.5 : 2}
            strokeOpacity={dim(s.key)}
            dot={chart.maxPoints <= dotLimit}
            activeDot={{ r: large ? 5 : 4, fill: s.color }}
            isAnimationActive={!large}
          />
        ))}
        {/* Маркер лучшей эпохи каждой кривой — та же точка, что в сводке оверлея. */}
        {large &&
          series.map((s) => {
            const { best } = curveStats(chart.rows, s.key, !!meta?.lowerIsBetter);
            if (!best) return null;
            return (
              <ReferenceDot
                key={`best-${s.key}`}
                x={best.epoch}
                y={best.value}
                r={4.5}
                fill={s.color}
                stroke="none"
                opacity={dim(s.key)}
              />
            );
          })}
        {/* Brush — приближение диапазона эпох перетаскиванием краёв полосы. */}
        {large && (
          <Brush
            dataKey="epoch"
            height={32}
            travellerWidth={9}
            traveller={({ x, y, width, height: h }) => (
              <rect x={x} y={y} width={width} height={h} rx={3} className="mcmp-brush-traveller" />
            )}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
};

/**
 * Сетка графиков с пользовательским порядком (drag and drop за ручку
 * в заголовке; порядок хранится в localStorage по storageKey) и кнопкой
 * разворота графика в zoom-оверлей.
 */
export const CurveChartGrid: React.FC<{
  items: CurveGridItem[];
  storageKey: string;
  onZoom: (name: string) => void;
}> = ({ items, storageKey, onZoom }) => {
  const chartNames = React.useMemo(() => items.map((item) => item.chart.name), [items]);
  const reorder = useDragReorder(chartNames, storageKey);
  const orderedItems = reorder.orderedKeys.map(
    (name) => items.find((item) => item.chart.name === name)!,
  );

  return (
    <div className="metrics-charts">
      {orderedItems.map(({ chart, series, checkpoints }) => (
        <div
          key={chart.name}
          className={`metrics-chart-block${
            reorder.draggedKey === chart.name ? ' mcmp-chart--dragging' : ''
          }`}
          {...reorder.itemProps(chart.name)}
        >
          <p className="metrics-chart-title mcmp-chart-title">
            <span
              className="mcmp-chart-grip"
              title="Перетащите, чтобы изменить порядок графиков"
              {...reorder.handleProps(chart.name)}
            >
              <i className={`fas ${ICONS.dragHandle}`} aria-hidden="true"></i>
            </span>
            {chart.name}
            <ChartLegend series={series} />
            <UiTooltip content="Развернуть график" className="mcmp-chart-zoom">
              <button
                className="icon-button small"
                onClick={() => onZoom(chart.name)}
                aria-label={`Развернуть график ${chart.name}`}
              >
                <i className={`fas ${ICONS.enlarge}`}></i>
              </button>
            </UiTooltip>
          </p>
          <CurveChartView
            chart={chart}
            series={series}
            checkpoints={checkpoints}
            height={180}
            dotLimit={40}
          />
        </div>
      ))}
    </div>
  );
};

/**
 * Увеличенный график в модальном оверлее: сводка по сериям (лучшее значение
 * с эпохой, финал, эпоха весов), brush для приближения диапазона эпох,
 * hover по легенде подсвечивает кривую. Закрывается по Escape и клику мимо;
 * скролл страницы на время оверлея блокируется.
 */
export const CurveZoomOverlay: React.FC<{
  /** Заголовок (имя метрики, при необходимости — с выборкой). */
  title: string;
  chart: EpochChart;
  series: CurveSeries[];
  checkpoints: CheckpointMark[];
  onClose: () => void;
}> = ({ title, chart, series, checkpoints, onClose }) => {
  // Hover по легенде — подсветка одной кривой. State живёт в оверлее
  // и исчезает вместе с ним — сброс при закрытии не нужен.
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const meta = getMetricMeta(chart.name);

  // Закрытие по Escape и блокировка скролла страницы, пока оверлей открыт.
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  return (
    <div className="mcmp-zoom-overlay" onClick={onClose}>
      <div
        className="mcmp-zoom-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`График ${title}`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="mcmp-zoom-head">
          <p className="metrics-chart-title mcmp-chart-title">
            {title}
            <ChartLegend series={series} onHoverSeries={setHoveredKey} />
          </p>
          <UiTooltip content="Закрыть">
            <button className="icon-button" onClick={onClose} aria-label="Закрыть">
              <i className={`fas ${ICONS.close}`}></i>
            </button>
          </UiTooltip>
        </header>
        {/* Сводка по кривым: лучшее значение с эпохой, финал, эпоха весов. */}
        <div className="mcmp-zoom-stats">
          {series.map((s) => {
            const { best, final } = curveStats(chart.rows, s.key, !!meta?.lowerIsBetter);
            if (!best || !final) return null;
            const bestQuality = getMetricQuality(meta, best.value);
            const showCkpt = s.checkpointEpoch != null && s.checkpointEpoch <= chart.maxPoints;
            return (
              <div key={s.key} className="mcmp-zoom-stat" style={{ borderLeftColor: s.color }}>
                <p className="mcmp-zoom-stat-label">
                  <span className="mcmp-side-dot" style={{ background: s.color }} />
                  {s.label}
                </p>
                <div className="mcmp-zoom-stat-values">
                  <span className="mcmp-zoom-stat-item">
                    <span className="mcmp-zoom-stat-key">
                      {meta?.lowerIsBetter ? 'минимум' : 'максимум'}
                    </span>
                    <b className={bestQuality ? `class-report-score--${bestQuality}` : undefined}>
                      {formatMetricValue(best.value)}
                    </b>
                    <span className="mcmp-zoom-stat-sub">эпоха {best.epoch}</span>
                  </span>
                  <span className="mcmp-zoom-stat-item">
                    <span className="mcmp-zoom-stat-key">финал</span>
                    <b>{formatMetricValue(final.value)}</b>
                    <span className="mcmp-zoom-stat-sub">эпоха {final.epoch}</span>
                  </span>
                  {showCkpt && (
                    <span className="mcmp-zoom-stat-item">
                      <span className="mcmp-zoom-stat-key">веса</span>
                      <b>эпоха {s.checkpointEpoch}</b>
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mcmp-zoom-chart">
          <CurveChartView
            chart={chart}
            series={series}
            checkpoints={checkpoints}
            height="100%"
            dotLimit={80}
            large
            highlightKey={hoveredKey}
          />
        </div>
        <p className="mcmp-zoom-note">
          * Потяните края полосы под графиком, чтобы приблизить диапазон эпох; пунктирная
          вертикаль — эпоха сохранённых весов модели.
        </p>
      </div>
    </div>
  );
};
