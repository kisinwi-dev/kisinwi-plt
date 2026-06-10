import React from 'react';
import { ICONS } from '../../../constants/icons';
import { Tooltip } from '../../common/Tooltip';
import type { DistributionComparison, SplitDriftInfo } from '../../../types/datasetComparison';
import { SPLIT_LABELS, orderSplitKeys } from '../statsShared';
import { DRIFT_LEVEL_LABELS, DRIFT_TOOLTIP, JS_TOOLTIP, PSI_TOOLTIP, driftTone } from './driftUtils';

interface Props {
  distribution: DistributionComparison;
}

interface DriftChipProps {
  metric: string;
  /** Пояснение метрики — тултип на чипе. */
  hint: string;
  value: number | null;
  level: SplitDriftInfo['js_level'];
}

const DriftChip: React.FC<DriftChipProps> = ({ metric, hint, value, level }) => {
  if (value === null || level === null) {
    return (
      <Tooltip content={hint} className="vstats-chip vstats-hint">
        {metric}: нет данных
      </Tooltip>
    );
  }
  return (
    <Tooltip content={hint} className={`vstats-chip vstats-hint vcmp-chip--${driftTone(level)}`}>
      <i className={`fas ${ICONS.drift}`}></i>
      {metric}: {value.toFixed(3)} — {DRIFT_LEVEL_LABELS[level]}
    </Tooltip>
  );
};

/** Распределение классов: drift-метрики (JS/PSI) и изменения состава классов по сплитам. */
const CompareDistributionSection: React.FC<Props> = ({ distribution }) => {
  const splitKeys = orderSplitKeys(
    Array.from(new Set([
      ...Object.keys(distribution.class_changes),
      ...Object.keys(distribution.drift),
    ])),
  );

  return (
    <section className="detail-section vcmp-section">
      <h3 className="detail-section-title">
        <i className={`fas ${ICONS.drift}`}></i> Распределение классов
        <Tooltip content={DRIFT_TOOLTIP}>
          <i className={`fas ${ICONS.info} vcmp-title-info`}></i>
        </Tooltip>
      </h3>

      {splitKeys.length === 0 ? (
        <p className="vstats-empty">Нет данных о распределении классов.</p>
      ) : (
        <div className="vcmp-split-blocks">
          {splitKeys.map(key => {
            const drift = distribution.drift[key];
            const changes = distribution.class_changes[key];
            const composition = changes
              && (changes.added_classes.length > 0 || changes.removed_classes.length > 0);
            return (
              <div key={key} className="vcmp-split-block">
                <h5 className="vstats-subtitle">{SPLIT_LABELS[key] ?? key}</h5>
                {drift && (
                  <div className="vstats-chips">
                    <DriftChip metric="JS" hint={JS_TOOLTIP} value={drift.js_divergence} level={drift.js_level} />
                    <DriftChip metric="PSI" hint={PSI_TOOLTIP} value={drift.psi} level={drift.psi_level} />
                  </div>
                )}
                {changes && composition && (
                  <div className="tag-list">
                    {changes.added_classes.map(name => (
                      <span key={name} className="tag vcmp-tag--added">+{name}</span>
                    ))}
                    {changes.removed_classes.map(name => (
                      <span key={name} className="tag vcmp-tag--removed">−{name}</span>
                    ))}
                    {changes.common_classes.length > 0 && (
                      <span className="vcmp-card-note">
                        {changes.common_classes.length} общих классов
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};

export default CompareDistributionSection;
