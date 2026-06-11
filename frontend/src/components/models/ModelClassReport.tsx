import React, { useEffect, useState } from 'react';
import { metricsService } from '../../services/metricsService';
import type { ClassReport } from '../../services/metricsService';
import { CollapseChevron, getDisclosureProps } from '../common/Collapse';
import { ICONS } from '../../constants/icons';

interface Props {
  modelId: string;
  // Отчёт появляется только после завершения обучения — до этого не запрашиваем.
  enabled: boolean;
}

const formatScore = (v: number) => v.toFixed(4);

// Интенсивность фона ячейки по доле от суммы строки confusion matrix.
const cellBackground = (value: number, rowTotal: number, isDiagonal: boolean) => {
  if (rowTotal === 0 || value === 0) return undefined;
  const share = value / rowTotal;
  const alpha = 0.08 + share * 0.55;
  // Диагональ (верные предсказания) — синяя, ошибки — красноватые, в духе SPLIT_COLORS.
  return isDiagonal ? `rgba(94, 138, 177, ${alpha})` : `rgba(177, 94, 107, ${alpha})`;
};

/**
 * Отчёт по классам на test-выборке из сервиса metrics: confusion matrix
 * и per-class precision/recall/f1/support. Появляется после завершения
 * обучения; если отчёта нет (404, старые модели) — не рендерится.
 */
const ModelClassReport: React.FC<Props> = ({ modelId, enabled }) => {
  const [report, setReport] = useState<ClassReport | null>(null);
  const [open, setOpen] = useState(false);

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

  return (
    <div className="metrics-report-details">
      <div
        className="metrics-report-summary"
        {...getDisclosureProps(open, () => setOpen((o) => !o))}
      >
        <CollapseChevron open={open} />
        <i className={`fas ${ICONS.classReport}`}></i> Отчёт по классам (test)
      </div>
      {open && (
        <div className="class-report">
          <div className="class-report-block">
            <p className="metrics-chart-title">Confusion matrix</p>
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
                          {value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="class-report-block">
            <p className="metrics-chart-title">Метрики по классам</p>
            <div className="class-report-table-wrap">
              <table className="class-report-table">
                <thead>
                  <tr>
                    <th>Класс</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {report.per_class.map((row) => (
                    <tr key={row.label}>
                      <th>{row.label}</th>
                      <td>{formatScore(row.precision)}</td>
                      <td>{formatScore(row.recall)}</td>
                      <td>{formatScore(row.f1)}</td>
                      <td>{row.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelClassReport;
