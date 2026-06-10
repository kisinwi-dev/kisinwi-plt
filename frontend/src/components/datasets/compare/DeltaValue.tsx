import React from 'react';

interface Props {
  delta: number;
  /** Процент изменения; null — базовое значение было 0; undefined — не показывать. */
  percentChange?: number | null;
  /** Форматирование абсолютной величины дельты (по умолчанию toLocaleString). */
  format?: (n: number) => string;
}

const defaultFormat = (n: number) => n.toLocaleString();

/**
 * Дельта «было → стало»: +N зелёным, −N красным, «без изменений» приглушённо.
 * Знак выводится отдельно от величины, минус — типографский.
 */
const DeltaValue: React.FC<Props> = ({ delta, percentChange, format = defaultFormat }) => {
  if (delta === 0) {
    return <span className="vcmp-delta vcmp-delta--zero">без изменений</span>;
  }
  const tone = delta > 0 ? 'up' : 'down';
  const sign = delta > 0 ? '+' : '−';
  return (
    <span className={`vcmp-delta vcmp-delta--${tone}`}>
      {sign}{format(Math.abs(delta))}
      {percentChange !== undefined && (
        <span className="vcmp-delta-pct">
          {percentChange === null
            ? '(новое)'
            : `(${percentChange > 0 ? '+' : '−'}${Math.abs(percentChange).toFixed(1)}%)`}
        </span>
      )}
    </span>
  );
};

export default DeltaValue;
