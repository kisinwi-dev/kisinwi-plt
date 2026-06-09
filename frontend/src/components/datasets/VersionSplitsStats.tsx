import React from 'react';
import { ICONS } from '../../constants/icons';
import type { VersionSplitsResponse } from '../../types/dataset';

interface Props {
  stats: VersionSplitsResponse;
  onClose: () => void;
}

const SPLIT_LABELS: Record<string, string> = {
  train: 'Train',
  val: 'Val',
  test: 'Test',
};

const VersionSplitsStats: React.FC<Props> = ({ stats, onClose }) => {
  const splitKeys = Object.keys(stats.splits_summary);

  return (
    <div className="version-splits-stats">
      <div className="splits-stats-header">
        <span className="splits-stats-title">Статистика сплитов</span>
        <button className="icon-button small" onClick={onClose} title="Скрыть">
          <i className={`fas ${ICONS.close}`}></i>
        </button>
      </div>
      <div className="splits-overall">
        <span className="splits-overall-label">
          Баланс классов:
          <span className={stats.overall_balance >= 0.7 ? 'balance-good' : 'balance-poor'}>
            {' '}{(stats.overall_balance * 100).toFixed(0)}%
            {' '}({stats.overall_balance >= 0.7 ? 'сбалансирован' : 'несбалансирован'})
          </span>
        </span>
        <div className="balance-bar">
          <div className="balance-bar-fill" style={{ width: `${stats.overall_balance * 100}%` }} />
        </div>
      </div>

      {splitKeys.length > 0 && (
        <div className="splits-overview">
          {splitKeys.map(key => {
            const split = stats.splits_summary[key];
            const sizeInfo = stats.image_size_stats?.[key];
            const label = SPLIT_LABELS[key] ?? key;
            return (
              <div key={key} className="split-block">
                <h6>{label}</h6>
                <div className="split-row">
                  <span>{split.total_samples.toLocaleString()} сэмплов</span>
                  <span>{split.num_classes} классов</span>
                </div>
                <div className="split-row">
                  <span className={split.is_balanced ? 'balance-good' : 'balance-poor'}>
                    {split.is_balanced ? '✓ сбалансирован' : '✗ несбалансирован'}
                  </span>
                </div>
                {split.class_distribution.length > 0 && (
                  <div className="split-classes">
                    <div className="split-class-header">
                      <span className="split-class-name">Класс</span>
                      <span className="split-class-count">Кол-во</span>
                      <span className="split-class-pct">%</span>
                    </div>
                    {split.class_distribution.slice(0, 5).map(cd => (
                      <div key={cd.class_name} className="split-class-row">
                        <span className="split-class-name">{cd.class_name}</span>
                        <span className="split-class-count">{cd.count.toLocaleString()}</span>
                        <span className="split-class-pct">{cd.percentage?.toFixed(1)}%</span>
                      </div>
                    ))}
                    {split.class_distribution.length > 5 && (
                      <span className="split-more">+{split.class_distribution.length - 5} ещё</span>
                    )}
                  </div>
                )}
                {sizeInfo && (
                  <div className="split-size-info">
                    <span>Размер: {sizeInfo.most_common_size}</span>
                    <span>Единообразие: {(sizeInfo.size_consistency * 100).toFixed(0)}%</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default VersionSplitsStats;
