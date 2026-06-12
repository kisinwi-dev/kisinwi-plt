import React from 'react';
import type { CompareMetric } from '../../services/metricsService';
import { ICONS } from '../../constants/icons';
import type { CompareSide } from './modelCompare';
import { metricLeaders } from './modelCompare';

interface Props {
  /** Стороны сравнения; первая — базовая. */
  sides: CompareSide[];
  /** Метрики сравнения на test. */
  metrics: CompareMetric[];
}

const SideChip: React.FC<{ side: CompareSide; role?: string }> = ({ side, role }) => (
  <span className="mcmp-summary-model">
    <span className="mcmp-side-dot" style={{ background: side.color }}></span>
    <span className="mcmp-summary-model-label">{side.label}</span>
    {role && <span className="mcmp-summary-model-role">{role}</span>}
  </span>
);

/**
 * Вердикт сравнения: кто лидирует и по скольким тестовым метрикам,
 * со списком метрик-побед каждой модели. Победа — лучшее значение метрики
 * с учётом ничьих (metricLeaders): при равенстве лучших значений победа
 * засчитывается всем лидерам, при равенстве у всех моделей — никому.
 */
const ModelCompareSummary: React.FC<Props> = ({ sides, metrics }) => {
  if (metrics.length === 0) return null;

  const leadersByMetric = metrics.map((m) => ({ metric: m.metric, leaders: metricLeaders(m) }));
  const winsBySide = sides.map((side) => ({
    side,
    wins: leadersByMetric
      .filter(({ leaders }) => leaders.includes(side.id))
      .map(({ metric }) => metric),
  }));
  const maxWins = Math.max(...winsBySide.map((w) => w.wins.length));
  const leaders = winsBySide.filter((w) => w.wins.length === maxWins && maxWins > 0);
  const leader = leaders.length === 1 ? leaders[0] : null;

  return (
    <section className="detail-section mcmp-summary">
      <div className="mcmp-summary-models">
        {sides.map((side, index) => (
          <React.Fragment key={side.id}>
            {index > 0 && (
              <i className={`fas ${ICONS.arrowRight} mcmp-summary-vs`} aria-hidden="true"></i>
            )}
            <SideChip side={side} role={index === 0 ? 'базовая' : undefined} />
          </React.Fragment>
        ))}
      </div>
      <p className="mcmp-summary-verdict">
        <i className={`fas ${ICONS.star}`} aria-hidden="true"></i>
        {leader ? (
          <>
            <strong style={{ color: leader.side.color }}>{leader.side.label}</strong>
            &nbsp;лучше по {leader.wins.length} из {metrics.length} метрик на тестовой выборке
          </>
        ) : maxWins > 0 ? (
          <>Паритет: по {maxWins} метрик у {leaders.length} моделей на тестовой выборке</>
        ) : (
          <>Полный паритет: значения всех общих метрик на тестовой выборке совпадают</>
        )}
      </p>
      <p className="mcmp-summary-basis">
        Сравнение — по метрикам на test: они измерены на сохранённых весах
        каждой модели — весах эпохи с лучшим значением loss на валидационной
        выборке (если в конфиге не задана другая early-stop-метрика).
      </p>
      <div className="mcmp-summary-wins">
        {winsBySide.map(
          ({ side, wins }) =>
            wins.length > 0 && (
              <span key={side.id} className="mcmp-summary-wins-row">
                <span className="mcmp-side-dot" style={{ background: side.color }}></span>
                лидирует: {wins.join(', ')}
              </span>
            ),
        )}
      </div>
    </section>
  );
};

export default ModelCompareSummary;
