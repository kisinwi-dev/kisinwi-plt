import React from 'react';
import type { ModelMetrics, Split } from '../../services/metricsService';
import { ICONS } from '../../constants/icons';
import { CollapseChevron, getDisclosureProps } from '../common/Collapse';
import { InfoHint } from '../common/Tooltip';
import type { CompareSide } from './modelCompare';
import { CurveChartGrid, CurveZoomOverlay } from './curveChart';
import { yDomain } from './curveChartData';
import type { CheckpointMark, CurveGridItem, CurveSeries, EpochChart } from './curveChartData';

interface Props {
  /** Стороны сравнения; первая — базовая. */
  sides: CompareSide[];
  /** Кривые есть только у train/val: test — одна точка. */
  split: Exclude<Split, 'test'>;
  /** Поэпоховые метрики каждой стороны (параллельно sides); null — метрик нет. */
  metricsBySide: Array<ModelMetrics | null>;
  status: 'loading' | 'error' | 'ready';
  onRetry: () => void;
  /** Раскрыта ли секция изначально (train сворачиваем — вторичен для сравнения). */
  defaultOpen: boolean;
}

const seriesOf = (metrics: ModelMetrics | null, split: Exclude<Split, 'test'>) => {
  const map = new Map<string, number[]>();
  for (const metric of metrics?.[split] ?? []) {
    if (metric.values.length > 0) map.set(metric.name, metric.values);
  }
  return map;
};

/**
 * Наложенные кривые обучения сравниваемых моделей: на каждую метрику,
 * по которой кривые есть минимум у двух моделей, — один график с линиями
 * цветов моделей и общей осью эпох. Метрики, имеющиеся лишь у одной модели,
 * перечисляются сноской.
 */
const ModelCompareCurves: React.FC<Props> = ({
  sides,
  split,
  metricsBySide,
  status,
  onRetry,
  defaultOpen,
}) => {
  const [open, setOpen] = React.useState(defaultOpen);
  // Сборка графиков — самая дорогая часть страницы (recharts): мемоизируем,
  // чтобы не пересчитывать при рендерах, не меняющих данные кривых.
  // Серия графика — модель: ключ m<индекс стороны>, цвет и пунктир чекпоинта
  // (эпоха сохранённых весов) — цвета модели.
  const { items, singleSide } = React.useMemo(() => {
    const seriesBySide = metricsBySide.map((m) => seriesOf(m, split));
    const metricNames = new Set<string>();
    for (const series of seriesBySide) {
      for (const name of series.keys()) metricNames.add(name);
    }

    const items: CurveGridItem[] = [];
    const singleSide: Array<{ name: string; side: CompareSide }> = [];
    for (const name of metricNames) {
      const sideIndexes = seriesBySide
        .map((series, index) => ({ index, values: series.get(name) }))
        .filter(({ values }) => values !== undefined && values.length > 1)
        .map(({ index }) => index);
      if (sideIndexes.length < 2) {
        const holders = seriesBySide
          .map((series, index) => (series.has(name) ? index : -1))
          .filter((i) => i >= 0);
        if (holders.length === 1) singleSide.push({ name, side: sides[holders[0]] });
        continue;
      }
      const maxPoints = Math.max(
        ...sideIndexes.map((index) => seriesBySide[index].get(name)!.length),
      );
      const rows = Array.from({ length: maxPoints }, (_, i) => {
        const row: EpochChart['rows'][number] = { epoch: i + 1 };
        for (const index of sideIndexes) {
          const value = seriesBySide[index].get(name)![i];
          if (value !== undefined) row[`m${index}`] = value;
        }
        return row;
      });
      const chart: EpochChart = {
        name,
        rows,
        maxPoints,
        domain: yDomain(rows, sideIndexes.map((index) => `m${index}`)),
      };
      const series: CurveSeries[] = sideIndexes.map((index) => ({
        key: `m${index}`,
        label: sides[index].label,
        color: sides[index].color,
        checkpointEpoch: metricsBySide[index]?.checkpoint?.epoch,
      }));
      const checkpoints: CheckpointMark[] = sideIndexes.flatMap((index) => {
        const ckpt = metricsBySide[index]?.checkpoint;
        return ckpt && ckpt.epoch <= maxPoints
          ? [{ epoch: ckpt.epoch, color: sides[index].color, seriesKey: `m${index}` }]
          : [];
      });
      items.push({ chart, series, checkpoints });
    }
    return { items, singleSide };
  }, [metricsBySide, split, sides]);

  // Увеличенный график в оверлее: имя метрики или null.
  const [zoomedName, setZoomedName] = React.useState<string | null>(null);
  const zoomedItem = items.find((item) => item.chart.name === zoomedName) ?? null;

  const body = () => {
    if (status === 'loading') {
      return (
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка кривых обучения…
        </div>
      );
    }
    if (status === 'error') {
      return (
        <div className="metrics-charts-placeholder metrics-charts-placeholder--error">
          <i className={`fas ${ICONS.warning}`}></i> Не удалось загрузить данные по эпохам.
          <button className="button secondary small" onClick={onRetry}>
            Повторить
          </button>
        </div>
      );
    }
    if (metricsBySide.every((m) => m === null)) {
      return (
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.metrics}`}></i> Данные по эпохам отсутствуют.
        </div>
      );
    }

    if (items.length === 0) {
      return (
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.metrics}`}></i> На выборке {split} нет общих кривых для наложения.
        </div>
      );
    }

    // Сноска: метрики, кривая которых есть только у одной модели.
    const notes = new Map<string, string[]>();
    for (const { name, side } of singleSide) {
      notes.set(side.label, [...(notes.get(side.label) ?? []), name]);
    }

    return (
      <>
        {/* Порядок графиков пользовательский — отдельный для train и val. */}
        <CurveChartGrid
          items={items}
          storageKey={`mcmp-curves-order:${split}`}
          onZoom={setZoomedName}
        />
        {notes.size > 0 && (
          <p className="mcmp-curves-note">
            {Array.from(notes.entries()).map(([label, names]) => (
              <span key={label}>Только у {label}: {names.join(', ')}. </span>
            ))}
          </p>
        )}
      </>
    );
  };

  const hint =
    split === 'val'
      ? 'Как метрика на валидации менялась по эпохам у каждой модели. Пунктирная вертикаль цвета модели — эпоха, с которой взяты сохранённые веса (лучший loss на валидационной выборке, если в конфиге не задана другая early-stop-метрика): именно эти веса тестируются и сравниваются в таблице выше. Резкие скачки и ухудшение к концу — признак нестабильного обучения.'
      : 'Как метрика на обучающей выборке менялась по эпохам. Пунктирная вертикаль — эпоха сохранённых весов. Сильный разрыв между train- и val-кривыми (train растёт, val стоит или ухудшается) — признак переобучения.';
  const desc =
    split === 'val'
      ? 'Динамика метрик на валидационной выборке. Пунктирная вертикаль — эпоха, веса которой сохранены и тестировались.'
      : 'Динамика метрик на обучающей выборке.';

  return (
    <section className="detail-section">
      <div
        className="metrics-report-summary"
        {...getDisclosureProps(open, () => setOpen((o) => !o))}
      >
        <CollapseChevron open={open} />
        <i className={`fas ${ICONS.metrics}`}></i> Кривые обучения ({split})
        <InfoHint text={hint} />
      </div>
      {open && (
        <>
          <p className="mcmp-section-desc">{desc}</p>
          {body()}
        </>
      )}
      {zoomedItem && (
        <CurveZoomOverlay
          title={`${zoomedItem.chart.name} (${split})`}
          chart={zoomedItem.chart}
          series={zoomedItem.series}
          checkpoints={zoomedItem.checkpoints}
          onClose={() => setZoomedName(null)}
        />
      )}
    </section>
  );
};

// memo: страница часто перерендеривается (загрузки, stale-refresh) —
// без мемоизации все recharts-графики отрисовывались бы заново каждый раз.
export default React.memo(ModelCompareCurves);
