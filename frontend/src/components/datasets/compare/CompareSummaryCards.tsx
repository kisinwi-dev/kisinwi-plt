import React from 'react';
import { ICONS } from '../../../constants/icons';
import { Tooltip } from '../../common/Tooltip';
import type { VersionComparisonResponse } from '../../../types/datasetComparison';
import { BALANCE_TOOLTIP, StatCard, balanceTone } from '../statsShared';
import { DRIFT_LEVEL_LABELS, DRIFT_TOOLTIP, driftTone, worstDriftLevel } from './driftUtils';
import DeltaValue from './DeltaValue';
import { FROM_COLOR, TO_COLOR } from './DeltaBarList';

interface Props {
  comparison: VersionComparisonResponse;
  fromName: string;
  toName: string;
}

/**
 * Сводка сравнения: строка направления «базовая → сравниваемая» (цвета точек —
 * те же, что в легендах бар-списков) и карточки: образцы, файлы, дрейф, баланс.
 */
const CompareSummaryCards: React.FC<Props> = ({ comparison, fromName, toName }) => {
  const { counts, distribution, balance, files } = comparison;
  const drift = worstDriftLevel(distribution.drift);
  const fromBalance = balance.overall_balance.from_value;
  const toBalance = balance.overall_balance.to_value;

  return (
    <div className="vcmp-summary">
      <div className="vcmp-direction">
        <span className="vstats-legend-item">
          <span className="vstats-legend-dot" style={{ background: FROM_COLOR }} />
          {fromName}
          <span className="vcmp-direction-role">базовая</span>
        </span>
        <i className={`fas ${ICONS.arrowRight} vcmp-direction-arrow`}></i>
        <span className="vstats-legend-item">
          <span className="vstats-legend-dot" style={{ background: TO_COLOR }} />
          {toName}
          <span className="vcmp-direction-role">сравниваемая</span>
        </span>
      </div>

      <div className="vstats-cards">
        <StatCard
          icon={ICONS.samples}
          label="Образцов"
          hint="Общее число изображений: сколько было в базовой версии → сколько стало в сравниваемой."
          value={
            <span className="vcmp-card-was-now">
              {counts.num_samples.from_value.toLocaleString()}
              {' → '}
              {counts.num_samples.to_value.toLocaleString()}
            </span>
          }
        >
          <DeltaValue
            delta={counts.num_samples.delta}
            percentChange={counts.num_samples.percent_change}
          />
        </StatCard>

        <StatCard
          icon={ICONS.file}
          label="Файлы"
          hint="Изменения набора файлов относительно базовой версии: сколько добавлено и сколько удалено."
          value={
            <span className="vcmp-files-counts">
              <span className="vcmp-delta--up">+{files.added_count.toLocaleString()}</span>
              {' / '}
              <span className="vcmp-delta--down">−{files.removed_count.toLocaleString()}</span>
            </span>
          }
        >
          <span className="vcmp-card-note">{files.common_count.toLocaleString()} без изменений</span>
        </StatCard>

        <StatCard
          icon={ICONS.drift}
          label="Дрейф распределения"
          hint={`${DRIFT_TOOLTIP} Показан худший уровень среди всех сплитов.`}
          value={
            drift === null ? (
              '—'
            ) : (
              <Tooltip content={DRIFT_TOOLTIP} className={`vcmp-drift-value vcmp-drift-value--${driftTone(drift)}`}>
                {DRIFT_LEVEL_LABELS[drift]}
              </Tooltip>
            )
          }
        >
          <span className="vcmp-card-note">худший уровень по сплитам</span>
        </StatCard>

        <StatCard
          icon={ICONS.balance}
          label="Баланс классов"
          hint={BALANCE_TOOLTIP}
          value={
            <span className="vcmp-card-was-now">
              {(fromBalance * 100).toFixed(0)}%
              {' → '}
              <span className={`vstats-balance--${balanceTone(toBalance)}`}>
                {(toBalance * 100).toFixed(0)}%
              </span>
            </span>
          }
        >
          <DeltaValue
            delta={balance.overall_balance.delta * 100}
            format={n => `${n.toFixed(1)} п.п.`}
          />
        </StatCard>
      </div>
    </div>
  );
};

export default CompareSummaryCards;
