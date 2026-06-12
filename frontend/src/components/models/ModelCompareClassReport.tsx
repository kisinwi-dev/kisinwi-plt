import React, { useEffect, useState } from 'react';
import { metricsService } from '../../services/metricsService';
import type { ClassReport, PerClassMetrics } from '../../services/metricsService';
import { CollapseChevron, getDisclosureProps } from '../common/Collapse';
import { Tooltip, InfoHint } from '../common/Tooltip';
import { ICONS } from '../../constants/icons';
import { getMetricMeta, getMetricQuality } from '../../constants/metrics';
import type { CompareSide } from './modelCompare';
import { displayEqual } from './modelCompare';
import CompareDeltaChip from './CompareDeltaChip';
import CompareSideHeader from './CompareSideHeader';

interface Props {
  /** Стороны сравнения; первая — базовая (от неё считаются дельты). */
  sides: CompareSide[];
}

type PerClassMetricName = 'precision' | 'recall' | 'f1';
const PER_CLASS_METRICS: PerClassMetricName[] = ['precision', 'recall', 'f1'];

type FetchState =
  | { key: string; status: 'ready'; data: Array<ClassReport | null> }
  | { key: string; status: 'error' }
  | null;

/**
 * Значение метрики класса: число, окрашенное по порогам качества, со звездой
 * лидера строки и мини-баром величины (метрики per-class ∈ [0, 1]) —
 * бар даёт быстрое визуальное сравнение колонок без чтения цифр.
 */
const ScoreCell: React.FC<{
  metric: string;
  value: number;
  isBest: boolean;
  /** Сколько моделей делят лучшее значение (для текста tooltip при ничьей). */
  leadersCount: number;
}> = ({ metric, value, isBest, leadersCount }) => {
  const quality = getMetricQuality(getMetricMeta(metric), value);
  return (
    <>
      <span className="mcmp-cell-value">
        {isBest && (
          <Tooltip
            content={
              leadersCount > 1
                ? `Лучшее значение по классу — поровну у ${leadersCount} моделей`
                : 'Лучшее значение по классу среди сравниваемых моделей'
            }
          >
            <i className={`fas ${ICONS.star} mcmp-best-icon`} aria-hidden="true"></i>
          </Tooltip>
        )}
        <span className={quality ? `class-report-score--${quality}` : undefined}>
          {value.toFixed(4)}
        </span>
      </span>
      <span className="mcmp-class-bar" aria-hidden="true">
        <span
          className={`mcmp-class-bar-fill${quality ? ` mcmp-class-bar-fill--${quality}` : ''}`}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        ></span>
      </span>
    </>
  );
};

/**
 * Лидеры строки класса с учётом ничьих (та же логика, что metricLeaders
 * сводной таблицы): индексы моделей, чьё значение неотличимо от лучшего
 * с точностью отображения; все равны — лидера нет.
 */
const rowLeaders = (values: number[]): number[] => {
  const best = Math.max(...values);
  const leaders = values
    .map((_, index) => index)
    .filter((index) => displayEqual(values[index], best));
  return leaders.length === values.length ? [] : leaders;
};

/**
 * Сравнение моделей по классам на test: для каждого общего класса — значения
 * выбранной метрики (precision/recall/f1) всех моделей с дельтой от базовой.
 * Сворачиваемая подсекция внутри «Метрики (test)» (отчёт по классам — те же
 * тестовые метрики, детализированные по классам); отчёты грузятся лениво
 * при первом раскрытии.
 */
const ModelCompareClassReport: React.FC<Props> = ({ sides }) => {
  const [open, setOpen] = useState(false);
  const [metric, setMetric] = useState<PerClassMetricName>('f1');
  const [state, setState] = useState<FetchState>(null);
  const [retryKey, setRetryKey] = useState(0);

  const key = sides.map((s) => s.id).join('|');

  useEffect(() => {
    if (!open) return;
    if (state && state.key === key && state.status === 'ready') return;
    let cancelled = false;
    Promise.all(sides.map((side) => metricsService.getClassReport(side.id)))
      .then((data) => {
        if (!cancelled) setState({ key, status: 'ready', data });
      })
      .catch(() => {
        if (!cancelled) setState({ key, status: 'error' });
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, key, retryKey]);

  const current = state && state.key === key ? state : null;

  const body = () => {
    if (!current) {
      return (
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка отчётов по классам…
        </div>
      );
    }
    if (current.status === 'error') {
      return (
        <div className="metrics-charts-placeholder metrics-charts-placeholder--error">
          <i className={`fas ${ICONS.warning}`}></i> Не удалось загрузить отчёты по классам.
          <button
            className="button secondary small"
            onClick={() => { setState(null); setRetryKey((k) => k + 1); }}
          >
            Повторить
          </button>
        </div>
      );
    }

    const reports = current.data;
    const withoutReport = sides.filter((_, index) => !reports[index]);
    if (withoutReport.length > 0) {
      return (
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.info}`}></i>
          Отчёт по классам недоступен: {withoutReport.map((s) => s.label).join(', ')}.
          Он появляется после завершения обучения.
        </div>
      );
    }

    const byLabel = reports.map(
      (report) => new Map(report!.per_class.map((row) => [row.label, row])),
    );
    const common = reports[0]!.labels.filter((label) =>
      byLabel.every((map) => map.has(label)),
    );

    if (common.length === 0) {
      return (
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.info}`}></i> У моделей нет общих классов — сравнение невозможно.
        </div>
      );
    }

    // Сноска: классы, отсутствующие хотя бы у одной модели.
    const partial = new Map<string, string[]>();
    reports.forEach((report, index) => {
      for (const label of report!.labels) {
        if (!common.includes(label)) {
          partial.set(sides[index].label, [...(partial.get(sides[index].label) ?? []), label]);
        }
      }
    });

    const supportCell = (rows: PerClassMetrics[]) => {
      const supports = rows.map((r) => r.support);
      return supports.every((s) => s === supports[0]) ? supports[0] : supports.join(' / ');
    };

    const rowsOf = (label: string) => byLabel.map((map) => map.get(label)!);

    return (
      <>
        <div className="class-report-block-head">
          <p className="metrics-chart-title">
            Метрика
            <InfoHint text="Какую метрику сравнивать по классам. Значения подсвечены по качеству, дельта — от базовой модели." />
          </p>
          <div className="view-toggle view-toggle--compact" role="group" aria-label="Метрика по классам">
            {PER_CLASS_METRICS.map((name) => (
              <Tooltip key={name} content={getMetricMeta(name)?.description}>
                <button
                  type="button"
                  className={`view-toggle-btn${metric === name ? ' active' : ''}`}
                  onClick={() => setMetric(name)}
                >
                  {name}
                </button>
              </Tooltip>
            ))}
          </div>
        </div>
        <div className="class-report-table-wrap mcmp-table-wrap mcmp-class-table-wrap">
          <table className="class-report-table mcmp-class-table">
            <thead>
              <tr>
                <th>Класс</th>
                {sides.map((side, index) => (
                  <CompareSideHeader key={side.id} side={side} isBase={index === 0} />
                ))}
                <th>
                  <Tooltip content="Сколько примеров класса было в test-выборке (по моделям, если различается).">
                    <span>Support</span>
                  </Tooltip>
                </th>
              </tr>
            </thead>
            <tbody>
              {common.map((label) => {
                const rows = rowsOf(label);
                const baseValue = rows[0][metric];
                const leaders = rowLeaders(rows.map((row) => row[metric]));
                return (
                  <tr key={label}>
                    <th>{label}</th>
                    {rows.map((row, index) => (
                      <td
                        key={sides[index].id}
                        className={leaders.includes(index) ? 'mcmp-cell--best' : undefined}
                      >
                        <ScoreCell
                          metric={metric}
                          value={row[metric]}
                          isBest={leaders.includes(index)}
                          leadersCount={leaders.length}
                        />
                        {index > 0 && (
                          <span className="mcmp-cell-delta">
                            <CompareDeltaChip
                              base={baseValue}
                              other={row[metric]}
                              higherIsBetter
                              showPercent={false}
                            />
                          </span>
                        )}
                      </td>
                    ))}
                    <td className="mcmp-class-support">{supportCell(rows)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {partial.size > 0 && (
          <p className="mcmp-curves-note">
            {Array.from(partial.entries()).map(([label, names]) => (
              <span key={label}>Только у {label}: {names.join(', ')}. </span>
            ))}
          </p>
        )}
      </>
    );
  };

  return (
    <div className="mcmp-class-report">
      <div
        className="metrics-report-summary"
        {...getDisclosureProps(open, () => setOpen((o) => !o))}
      >
        <CollapseChevron open={open} />
        <i className={`fas ${ICONS.classReport}`}></i> Сравнение по классам
        <InfoHint text="Те же тестовые метрики, детализированные по классам: какие классы каждая модель распознаёт лучше. Полезно, когда общие метрики близки — модели могут ошибаться на разных классах." />
      </div>
      {open && body()}
    </div>
  );
};

export default ModelCompareClassReport;
