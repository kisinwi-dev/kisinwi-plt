import React from 'react';
// Общие примитивы статистики версий: используются модалкой статистики
// (VersionStatsModal) и страницей сравнения версий (DatasetCompare).
// Классы vstats-* живут в VersionStatsModal.css (CSS в Vite глобальный).
import { ICONS } from '../../constants/icons';
import { Tooltip } from '../common/Tooltip';
import './VersionStatsModal.css';

// eslint-disable-next-line react-refresh/only-export-components
export const SPLIT_LABELS: Record<string, string> = {
  train: 'Train',
  val: 'Val',
  test: 'Test',
};

const SPLIT_ORDER = ['train', 'val', 'test'];

export const BALANCE_THRESHOLD = 0.7;

// Цвет сплита по его индексу в splitKeys; токены заданы в :root theme.css
// и не переопределяются темами.
// eslint-disable-next-line react-refresh/only-export-components
export const SPLIT_COLORS = ['var(--color-accent)', 'var(--color-info)', 'var(--color-warning)'];

// Известные сплиты — в фиксированном порядке, неизвестные ключи — после них.
// eslint-disable-next-line react-refresh/only-export-components
export const orderSplitKeys = (keys: string[]): string[] =>
  [...keys].sort((a, b) => {
    const ia = SPLIT_ORDER.indexOf(a);
    const ib = SPLIT_ORDER.indexOf(b);
    return (ia === -1 ? SPLIT_ORDER.length : ia) - (ib === -1 ? SPLIT_ORDER.length : ib);
  });

// eslint-disable-next-line react-refresh/only-export-components
export const balanceTone = (ratio: number) => (ratio >= BALANCE_THRESHOLD ? 'good' : 'poor');

export const BALANCE_TOOLTIP =
  'Отношение размера самого малочисленного класса к самому многочисленному: '
  + '100% — изображений во всех классах поровну, '
  + `ниже ${BALANCE_THRESHOLD * 100}% — заметный дисбаланс.`;

interface StatCardProps {
  icon: string;
  label: string;
  value: React.ReactNode;
  tone?: 'good' | 'poor';
  /** Пояснение метрики — info-иконка с тултипом. */
  hint?: string;
  children?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({ icon, label, value, tone, hint, children }) => (
  <div className="vstats-card">
    <span className="vstats-card-label">
      <i className={`fas ${icon}`}></i> {label}
      {hint && (
        <Tooltip content={hint}>
          <i className={`fas ${ICONS.info} vstats-hint`}></i>
        </Tooltip>
      )}
    </span>
    <span className={`vstats-card-value${tone ? ` vstats-balance--${tone}` : ''}`}>{value}</span>
    {children}
  </div>
);

interface BarRowProps {
  label: string;
  count: number;
  maxCount: number;
  pct?: string;
}

// Ширина бара масштабируется от максимального count в списке, чтобы самая
// крупная строка занимала всю дорожку; настоящий процент показываем текстом.
export const BarRow: React.FC<BarRowProps> = ({ label, count, maxCount, pct }) => (
  <div className="vstats-bar-row">
    <Tooltip content={label} className="vstats-bar-label">{label}</Tooltip>
    <span className="vstats-bar-track">
      <span
        className="vstats-bar-fill"
        style={{ width: `${maxCount > 0 ? (count / maxCount) * 100 : 0}%` }}
      />
    </span>
    <span className="vstats-bar-count">{count.toLocaleString()}</span>
    <span className="vstats-bar-pct">{pct ?? ''}</span>
  </div>
);
