import React, { useEffect, useState } from 'react';
import { metricsService } from '../../services/metricsService';
import type { ClassReport } from '../../services/metricsService';
import { CollapseChevron, getDisclosureProps } from '../common/Collapse';
import { Tooltip, InfoHint } from '../common/Tooltip';
import { ICONS } from '../../constants/icons';
import { getMetricMeta, getMetricQuality } from '../../constants/metrics';

// Подсказки для обычного пользователя: что значит каждая таблица и колонка.
const HINTS = {
  report:
    'Показывает, какие классы модель распознаёт хорошо, а какие путает между собой.',
  matrix:
    'Строка — каким класс был на самом деле, столбец — что предсказала модель. Синяя диагональ — верные ответы, красные ячейки — ошибки; чем ярче цвет, тем больше доля таких ответов.',
  toggleCounts: 'Показывать число примеров.',
  togglePercent: 'Показывать долю от всех примеров фактического класса (строки).',
  support: 'Сколько примеров этого класса было в test-выборке.',
  macroAvg:
    'Среднее по всем классам, каждый класс учитывается одинаково — независимо от числа его примеров.',
  accuracy: 'Доля верных предсказаний по всей test-выборке.',
  perClass:
    'Качество модели отдельно для каждого класса. Значения подсвечены цветом: зелёный — хорошо, жёлтый — средне, красный — слабо.',
} as const;

interface Props {
  modelId: string;
  // Отчёт появляется только после завершения обучения — до этого не запрашиваем.
  enabled: boolean;
}

type MatrixView = 'counts' | 'percent';

const formatScore = (v: number) => v.toFixed(4);
const formatShare = (value: number, rowTotal: number) =>
  rowTotal === 0 ? '—' : `${((value / rowTotal) * 100).toFixed(1)}%`;

// Интенсивность фона ячейки по доле от суммы строки confusion matrix.
const cellBackground = (value: number, rowTotal: number, isDiagonal: boolean) => {
  if (rowTotal === 0 || value === 0) return undefined;
  const share = value / rowTotal;
  const alpha = 0.08 + share * 0.55;
  // Диагональ (верные предсказания) — синяя, ошибки — красноватые, в духе SPLIT_COLORS.
  return isDiagonal ? `rgba(94, 138, 177, ${alpha})` : `rgba(177, 94, 107, ${alpha})`;
};

// Значение precision/recall/f1/accuracy с окраской по порогам качества метрики.
const ScoreCell: React.FC<{ metric: string; value: number }> = ({ metric, value }) => {
  const quality = getMetricQuality(getMetricMeta(metric), value);
  return (
    <span className={quality ? `class-report-score--${quality}` : undefined}>
      {formatScore(value)}
    </span>
  );
};

/**
 * Отчёт по классам на test-выборке из сервиса metrics: confusion matrix
 * (counts / нормализация по строке) и per-class precision/recall/f1/support
 * с итоговыми строками macro avg и accuracy. Появляется после завершения
 * обучения; если отчёта нет (404, старые модели) — не рендерится.
 */
const ModelClassReport: React.FC<Props> = ({ modelId, enabled }) => {
  const [report, setReport] = useState<ClassReport | null>(null);
  const [open, setOpen] = useState(true);
  const [matrixView, setMatrixView] = useState<MatrixView>('counts');

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    metricsService
      .getClassReport(modelId)
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [modelId, enabled]);

  if (!report) return null;

  const rowTotals = report.confusion_matrix.map((row) =>
    row.reduce((sum, v) => sum + v, 0),
  );

  // Итоги в духе sklearn classification_report: macro avg и accuracy.
  const classCount = report.per_class.length;
  const macro = {
    precision: report.per_class.reduce((s, r) => s + r.precision, 0) / (classCount || 1),
    recall: report.per_class.reduce((s, r) => s + r.recall, 0) / (classCount || 1),
    f1: report.per_class.reduce((s, r) => s + r.f1, 0) / (classCount || 1),
    support: report.per_class.reduce((s, r) => s + r.support, 0),
  };
  const total = rowTotals.reduce((s, v) => s + v, 0);
  const correct = report.confusion_matrix.reduce((s, row, i) => s + (row[i] ?? 0), 0);
  const accuracy = total > 0 ? correct / total : undefined;

  return (
    <div className="metrics-report-details">
      <div
        className="metrics-report-summary"
        {...getDisclosureProps(open, () => setOpen((o) => !o))}
      >
        <CollapseChevron open={open} />
        <i className={`fas ${ICONS.classReport}`}></i> Отчёт по классам (test)
        <InfoHint text={HINTS.report} />
      </div>
      {open && (
        <div className="class-report">
          <div className="class-report-block">
            <div className="class-report-block-head">
              <p className="metrics-chart-title">
                Confusion matrix
                <InfoHint text={HINTS.matrix} />
              </p>
              <div
                className="view-toggle view-toggle--compact"
                role="group"
                aria-label="Формат значений матрицы"
              >
                {(['counts', 'percent'] as const).map((view) => (
                  <Tooltip
                    key={view}
                    content={view === 'counts' ? HINTS.toggleCounts : HINTS.togglePercent}
                  >
                    <button
                      type="button"
                      className={`view-toggle-btn${matrixView === view ? ' active' : ''}`}
                      onClick={() => setMatrixView(view)}
                    >
                      {view === 'counts' ? 'N' : '%'}
                    </button>
                  </Tooltip>
                ))}
              </div>
            </div>
            <div className="class-report-table-wrap">
              <table className="class-report-table class-report-matrix">
                <thead>
                  <tr>
                    <th className="class-report-corner">факт \ прогноз</th>
                    {report.labels.map((label) => (
                      <th key={label}>{label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {report.confusion_matrix.map((row, i) => (
                    <tr key={report.labels[i]}>
                      <th>{report.labels[i]}</th>
                      {row.map((value, j) => (
                        <td
                          key={report.labels[j]}
                          style={{ background: cellBackground(value, rowTotals[i], i === j) }}
                        >
                          <Tooltip
                            content={`факт ${report.labels[i]}, прогноз ${report.labels[j]}: ${value} (${formatShare(value, rowTotals[i])})`}
                          >
                            {matrixView === 'counts' ? value : formatShare(value, rowTotals[i])}
                          </Tooltip>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="class-report-block">
            <div className="class-report-block-head">
              <p className="metrics-chart-title">
                Метрики по классам
                <InfoHint text={HINTS.perClass} />
              </p>
            </div>
            <div className="class-report-table-wrap">
              <table className="class-report-table">
                <thead>
                  <tr>
                    <th>Класс</th>
                    <th>
                      <Tooltip content={getMetricMeta('precision')?.description}>Precision</Tooltip>
                    </th>
                    <th>
                      <Tooltip content={getMetricMeta('recall')?.description}>Recall</Tooltip>
                    </th>
                    <th>
                      <Tooltip content={getMetricMeta('f1')?.description}>F1</Tooltip>
                    </th>
                    <th>
                      <Tooltip content={HINTS.support}>Support</Tooltip>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {report.per_class.map((row) => (
                    <tr key={row.label}>
                      <th>{row.label}</th>
                      <td><ScoreCell metric="precision" value={row.precision} /></td>
                      <td><ScoreCell metric="recall" value={row.recall} /></td>
                      <td><ScoreCell metric="f1" value={row.f1} /></td>
                      <td>{row.support}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <th>
                      <Tooltip content={HINTS.macroAvg}>macro avg</Tooltip>
                    </th>
                    <td><ScoreCell metric="precision" value={macro.precision} /></td>
                    <td><ScoreCell metric="recall" value={macro.recall} /></td>
                    <td><ScoreCell metric="f1" value={macro.f1} /></td>
                    <td>{macro.support}</td>
                  </tr>
                  {accuracy !== undefined && (
                    <tr>
                      <th>
                        <Tooltip content={HINTS.accuracy}>accuracy</Tooltip>
                      </th>
                      <td></td>
                      <td></td>
                      <td><ScoreCell metric="accuracy" value={accuracy} /></td>
                      <td>{total}</td>
                    </tr>
                  )}
                </tfoot>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelClassReport;
