import React from 'react';
import { displayEqual } from './modelCompare';
import { formatMetricValue } from '../../utils/format';

interface Props {
  /** Значение базовой модели (от него считается дельта и процент). */
  base: number;
  /** Значение сравниваемой модели. */
  other: number;
  higherIsBetter: boolean;
  /** Показывать процент от базового значения (по умолчанию да). */
  showPercent?: boolean;
}

/**
 * Чип дельты «сравниваемая − базовая» в стандартном для KPI-индикаторов виде
 * «▲ +0.0123 (+1.2%)»: треугольник — направление изменения значения
 * (▲ выросло, ▼ упало), цвет — улучшение с учётом направления метрики
 * (зелёный — сравниваемая модель лучше, красный — хуже; для loss «лучше» —
 * это ▼). Цвет дублируется треугольником намеренно — смысл читается и без
 * различения цветов. Серый прочерк — значения совпадают. Стили — vcmp-delta
 * из DatasetCompare.css.
 */
const CompareDeltaChip: React.FC<Props> = ({ base, other, higherIsBetter, showPercent = true }) => {
  const delta = other - base;
  // Значения, неразличимые с точностью отображения, — ничья: чип «▲+0.0000»
  // рядом с двумя одинаковыми на вид числами читается как ошибка.
  if (displayEqual(base, other)) {
    return <span className="vcmp-delta vcmp-delta--zero">—</span>;
  }
  const improved = higherIsBetter ? delta > 0 : delta < 0;
  const sign = delta > 0 ? '+' : '';
  // Дельта на границе округления может схлопнуться в «0.0000», хотя значения
  // отображаются по-разному, — такие показываем с двумя значащими цифрами.
  const deltaText =
    Number(formatMetricValue(delta)) === 0 ? delta.toPrecision(2) : formatMetricValue(delta);
  const percent =
    showPercent && Math.abs(base) > 0
      ? `${sign}${((delta / Math.abs(base)) * 100).toFixed(1)}%`
      : null;
  return (
    <span
      className={`vcmp-delta ${improved ? 'vcmp-delta--up' : 'vcmp-delta--down'}`}
      title={`Δ от базовой модели: ${sign}${deltaText}${percent ? ` (${percent})` : ''}`}
    >
      <span className="mcmp-delta-dir" aria-hidden="true">{delta > 0 ? '▲' : '▼'}</span>
      {sign}
      {deltaText}
      {percent && <span className="vcmp-delta-pct">({percent})</span>}
    </span>
  );
};

export default CompareDeltaChip;
