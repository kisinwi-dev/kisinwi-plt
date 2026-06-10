import React, { useMemo, useState } from 'react';
import { ICONS } from '../../../constants/icons';
import { Tooltip } from '../../common/Tooltip';
import type { CountsComparison } from '../../../types/datasetComparison';
import { SPLIT_LABELS, orderSplitKeys } from '../statsShared';
import DeltaBarList from './DeltaBarList';

interface Props {
  counts: CountsComparison;
  fromName: string;
  toName: string;
}

/** Изменение количества изображений: по сплитам и по классам внутри сплита. */
const CompareCountsSection: React.FC<Props> = ({ counts, fromName, toName }) => {
  const [selectedSplit, setSelectedSplit] = useState<string | null>(null);

  const splitRows = useMemo(
    () =>
      orderSplitKeys(Object.keys(counts.per_split)).map(key => ({
        label: SPLIT_LABELS[key] ?? key,
        delta: counts.per_split[key],
      })),
    [counts.per_split],
  );

  const classSplitKeys = useMemo(
    () => orderSplitKeys(Object.keys(counts.per_class)),
    [counts.per_class],
  );

  // Активный таб — производное значение: выбранный сплит, если он есть, иначе первый.
  const activeSplit = selectedSplit && classSplitKeys.includes(selectedSplit)
    ? selectedSplit
    : classSplitKeys[0] ?? null;

  // Классы с наибольшим изменением — сверху.
  const classRows = useMemo(() => {
    if (!activeSplit) return [];
    return Object.entries(counts.per_class[activeSplit] ?? {})
      .map(([name, delta]) => ({ label: name, delta }))
      .sort((a, b) => Math.abs(b.delta.delta) - Math.abs(a.delta.delta));
  }, [counts.per_class, activeSplit]);

  return (
    <section className="detail-section vcmp-section">
      <h3 className="detail-section-title">
        <i className={`fas ${ICONS.samples}`}></i> Количество изображений
        <Tooltip content="Число изображений в каждом сплите и классе: базовая версия → сравниваемая. Δ — на сколько изменилось.">
          <i className={`fas ${ICONS.info} vcmp-title-info`}></i>
        </Tooltip>
      </h3>

      {(counts.added_splits.length > 0 || counts.removed_splits.length > 0) && (
        <div className="vstats-chips">
          {counts.added_splits.map(key => (
            <span key={key} className="vstats-chip vstats-chip--good">
              <i className={`fas ${ICONS.add}`}></i> сплит {SPLIT_LABELS[key] ?? key}
            </span>
          ))}
          {counts.removed_splits.map(key => (
            <span key={key} className="vstats-chip vstats-chip--poor">
              <i className={`fas ${ICONS.minus}`}></i> сплит {SPLIT_LABELS[key] ?? key}
            </span>
          ))}
        </div>
      )}

      {splitRows.length === 0 ? (
        <p className="vstats-empty">Нет данных о сплитах.</p>
      ) : (
        <DeltaBarList rows={splitRows} fromLabel={fromName} toLabel={toName} headLabel="Сплит" />
      )}

      {classSplitKeys.length > 0 && activeSplit && (
        <>
          <h5 className="vstats-subtitle">
            <i className={`fas ${ICONS.classes}`}></i> По классам
          </h5>
          {classSplitKeys.length > 1 && (
            <div className="vstats-tabs">
              {classSplitKeys.map(key => (
                <button
                  key={key}
                  className={`vstats-tab${key === activeSplit ? ' vstats-tab--active' : ''}`}
                  onClick={() => setSelectedSplit(key)}
                >
                  {SPLIT_LABELS[key] ?? key}
                </button>
              ))}
            </div>
          )}
          {classRows.length === 0 ? (
            <p className="vstats-empty">Нет данных о классах.</p>
          ) : (
            <DeltaBarList rows={classRows} fromLabel={fromName} toLabel={toName} headLabel="Класс" />
          )}
        </>
      )}
    </section>
  );
};

export default CompareCountsSection;
