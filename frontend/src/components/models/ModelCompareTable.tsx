import React from 'react';
import type { CompareMetric, CompareModelEntry } from '../../services/metricsService';
import { Tooltip } from '../common/Tooltip';
import { ICONS } from '../../constants/icons';
import { getMetricMeta, getMetricQuality } from '../../constants/metrics';
import type { CompareSide, WeightsInfo } from './modelCompare';
import { metricLeaders, primaryValue } from './modelCompare';
import { formatMetricValue } from '../../utils/format';
import CompareDeltaChip from './CompareDeltaChip';
import CompareSideHeader from './CompareSideHeader';

interface Props {
  /** Стороны сравнения; первая — базовая (от неё считаются дельты). */
  sides: CompareSide[];
  /** Метрики сравнения на test. */
  metrics: CompareMetric[];
  /** Происхождение весов каждой стороны (параллельно sides); null — чекпоинт не записан. */
  weightsBySide: Array<WeightsInfo | null>;
}

/**
 * Ячейка значения модели: тестовое значение (по нему считаются лидер
 * и дельты); для небазовых моделей — дельта от базовой.
 * Лидерство учитывает ничьи: при равенстве
 * лучших значений звезду получают все модели-лидеры, при равенстве у всех —
 * никто (leaders пуст).
 */
const ValueCell: React.FC<{
  metric: CompareMetric;
  entry: CompareModelEntry | undefined;
  baseEntry: CompareModelEntry | undefined;
  isBase: boolean;
  /** Модели-лидеры по метрике (с учётом ничьих). */
  leaders: string[];
}> = ({ metric, entry, baseEntry, isBase, leaders }) => {
  if (!entry) return <td>—</td>;
  const value = primaryValue(entry);
  const quality = getMetricQuality(getMetricMeta(metric.metric), value);
  const isBest = leaders.includes(entry.model_id);
  return (
    <td className={isBest ? 'mcmp-cell--best' : undefined}>
      <span className="mcmp-cell-value">
        {isBest && (
          <Tooltip
            content={
              leaders.length > 1
                ? `Лучшее значение на test — поровну у ${leaders.length} моделей`
                : 'Лучшее значение на test среди сравниваемых моделей'
            }
          >
            <i className={`fas ${ICONS.star} mcmp-best-icon`} aria-hidden="true"></i>
          </Tooltip>
        )}
        <span className={quality ? `class-report-score--${quality}` : undefined}>
          {formatMetricValue(value)}
        </span>
      </span>
      {!isBase && baseEntry && (
        <span className="mcmp-cell-delta">
          <CompareDeltaChip
            base={primaryValue(baseEntry)}
            other={value}
            higherIsBetter={metric.higher_is_better}
          />
        </span>
      )}
    </td>
  );
};

/** Подстрока шапки колонки: с какой эпохи взяты веса итоговой модели. */
const WeightsNote: React.FC<{ info: WeightsInfo | null }> = ({ info }) => {
  if (!info) {
    return (
      <Tooltip content="Модель обучена до ввода учёта чекпоинтов: trainer сохраняет веса эпохи с лучшим значением early-stop-метрики на валидационной выборке, но для этой модели эпоха не записана">
        <span className="mcmp-th-weights">эпоха весов неизвестна</span>
      </Tooltip>
    );
  }
  const total = info.totalEpochs ? `/${info.totalEpochs}` : '';
  return (
    <Tooltip content={`Какая эпоха обучения ушла в реестр и тестировалась: trainer сохраняет веса эпохи с лучшим значением ${info.metric} на валидационной выборке (val_${info.metric})`}>
      <span className="mcmp-th-weights">
        веса: эпоха {info.epoch}{total} — лучший {info.metric} на валидации
      </span>
    </Tooltip>
  );
};

/**
 * Сводная diff-таблица сравнения по test: строка — метрика, колонка — модель.
 * Лидер по метрике помечен звездой, у небазовых моделей — дельта от базовой;
 * в шапке колонки — эпоха, с которой взяты сохранённые веса.
 */
const ModelCompareTable: React.FC<Props> = ({ sides, metrics, weightsBySide }) => {
  const entryOf = (metric: CompareMetric, id: string) =>
    metric.models.find((m) => m.model_id === id);

  const base = sides[0];

  return (
    <div className="class-report-table-wrap mcmp-table-wrap">
      <table className="class-report-table mcmp-table">
        <thead>
          <tr>
            <th>Метрика</th>
            {sides.map((side, index) => (
              <CompareSideHeader key={side.id} side={side} isBase={index === 0}>
                <WeightsNote info={weightsBySide[index] ?? null} />
              </CompareSideHeader>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric) => {
            const meta = getMetricMeta(metric.metric);
            const baseEntry = entryOf(metric, base.id);
            const leaders = metricLeaders(metric);
            const direction = metric.higher_is_better ? 'Больше — лучше.' : 'Меньше — лучше.';
            return (
              <tr key={metric.metric}>
                <th>
                  <Tooltip content={meta ? `${meta.description} ${direction}` : direction}>
                    <span className="mcmp-metric-name">
                      {meta && <i className={`fas ${meta.icon}`} aria-hidden="true"></i>}
                      {metric.metric}
                      <span className="mcmp-direction">{metric.higher_is_better ? '↑' : '↓'}</span>
                    </span>
                  </Tooltip>
                </th>
                {sides.map((side, index) => (
                  <ValueCell
                    key={side.id}
                    metric={metric}
                    entry={entryOf(metric, side.id)}
                    baseEntry={baseEntry}
                    isBase={index === 0}
                    leaders={leaders}
                  />
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default ModelCompareTable;
