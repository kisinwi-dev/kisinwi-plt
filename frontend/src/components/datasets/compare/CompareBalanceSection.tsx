import React from 'react';
import { ICONS } from '../../../constants/icons';
import { Tooltip } from '../../common/Tooltip';
import type { BalanceComparison, ValueDelta } from '../../../types/datasetComparison';
import { BALANCE_TOOLTIP, SPLIT_LABELS, balanceTone, orderSplitKeys } from '../statsShared';
import DeltaValue from './DeltaValue';

interface Props {
  balance: BalanceComparison;
}

const BalanceRow: React.FC<{ label: string; delta: ValueDelta }> = ({ label, delta }) => (
  <div className="vcmp-balance-row">
    <Tooltip content={label} className="vstats-bar-label">{label}</Tooltip>
    <span className="vcmp-balance-values">
      {(delta.from_value * 100).toFixed(0)}% → {(delta.to_value * 100).toFixed(0)}%
    </span>
    <span className="vstats-card-bar">
      <span
        className={`vstats-card-bar-fill vstats-card-bar-fill--${balanceTone(delta.to_value)}`}
        style={{ width: `${Math.min(delta.to_value, 1) * 100}%` }}
      />
    </span>
    <span className="vcmp-delta-col">
      <DeltaValue delta={delta.delta * 100} format={n => `${n.toFixed(1)} п.п.`} />
    </span>
  </div>
);

/** Баланс классов: общий коэффициент и по сплитам, «было → стало» в процентах. */
const CompareBalanceSection: React.FC<Props> = ({ balance }) => (
  <section className="detail-section vcmp-section">
    <h3 className="detail-section-title">
      <i className={`fas ${ICONS.balance}`}></i> Баланс классов
      <Tooltip content={BALANCE_TOOLTIP}>
        <i className={`fas ${ICONS.info} vcmp-title-info`}></i>
      </Tooltip>
    </h3>
    <div className="vcmp-balance-list">
      <BalanceRow label="Общий" delta={balance.overall_balance} />
      {orderSplitKeys(Object.keys(balance.per_split)).map(key => (
        <BalanceRow key={key} label={SPLIT_LABELS[key] ?? key} delta={balance.per_split[key]} />
      ))}
    </div>
  </section>
);

export default CompareBalanceSection;
