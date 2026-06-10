import React from 'react';
import { Tooltip } from '../../common/Tooltip';
import type { ValueDelta } from '../../../types/datasetComparison';
import DeltaValue from './DeltaValue';

export interface DeltaBarRow {
  label: string;
  delta: ValueDelta;
}

interface Props {
  rows: DeltaBarRow[];
  /** Имя базовой версии для легенды. */
  fromLabel: string;
  /** Имя сравниваемой версии для легенды. */
  toLabel: string;
  /** Заголовок первой колонки (Класс / Формат / Размер …). */
  headLabel: string;
}

// Цвета пары «было/стало» — те же роли, что у сплитов в SplitComparePanel:
// базовая версия — info, сравниваемая — accent. Переиспользуются строкой
// направления сравнения в CompareSummaryCards.
export const FROM_COLOR = 'var(--color-info)';
export const TO_COLOR = 'var(--color-accent)';

/**
 * Список строк «было → стало»: по два бара на строку (базовая и сравниваемая
 * версии) в общем масштабе + колонка дельты. Паттерн SplitComparePanel.
 */
const DeltaBarList: React.FC<Props> = ({ rows, fromLabel, toLabel, headLabel }) => {
  if (rows.length === 0) {
    return <p className="vstats-empty">Нет данных.</p>;
  }

  // Общий масштаб баров для всего списка, чтобы сравнение было честным.
  const maxValue = Math.max(...rows.flatMap(r => [r.delta.from_value, r.delta.to_value]), 0);

  return (
    <div className="vcmp-bar-block">
      <div className="vstats-compare-legend">
        <span className="vstats-legend-item">
          <span className="vstats-legend-dot" style={{ background: FROM_COLOR }} />
          {fromLabel}
        </span>
        <span className="vstats-legend-item">
          <span className="vstats-legend-dot" style={{ background: TO_COLOR }} />
          {toLabel}
        </span>
      </div>
      <div className="vstats-bar-list">
        <div className="vcmp-row vstats-bar-head">
          <span className="vstats-bar-label">{headLabel}</span>
          <span className="vstats-bar-label">Кол-во</span>
          <Tooltip
            content="Δ — изменение: значение в сравниваемой версии минус значение в базовой."
            className="vcmp-delta-col vstats-hint"
          >
            Δ
          </Tooltip>
        </div>
        {rows.map(row => (
          <div key={row.label} className="vcmp-row">
            <Tooltip content={row.label} className="vstats-bar-label">{row.label}</Tooltip>
            <span className="vstats-compare-bars">
              {([
                [row.delta.from_value, FROM_COLOR],
                [row.delta.to_value, TO_COLOR],
              ] as const).map(([value, color], i) => (
                <span key={i} className="vcmp-line">
                  <span className="vstats-bar-track">
                    <span
                      className="vstats-bar-fill"
                      style={{
                        width: `${maxValue > 0 ? (value / maxValue) * 100 : 0}%`,
                        background: color,
                      }}
                    />
                  </span>
                  <span className="vstats-compare-pct">{value.toLocaleString()}</span>
                </span>
              ))}
            </span>
            <span className="vcmp-delta-col">
              <DeltaValue delta={row.delta.delta} percentChange={row.delta.percent_change} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DeltaBarList;
