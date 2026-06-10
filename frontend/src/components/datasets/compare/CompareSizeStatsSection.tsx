import React, { useMemo, useState } from 'react';
import { ICONS } from '../../../constants/icons';
import { CollapseChevron, getDisclosureProps } from '../../common/Collapse';
import { Tooltip } from '../../common/Tooltip';
import type { SizeStatsComparison } from '../../../types/datasetComparison';
import { SPLIT_LABELS, orderSplitKeys } from '../statsShared';
import DeltaBarList from './DeltaBarList';

interface Props {
  sizeStats: SizeStatsComparison;
  fromName: string;
  toName: string;
}

// Размеров WxH может быть много — по умолчанию показываем самые изменившиеся.
const TOP_SIZES_LIMIT = 20;

/** Форматы изображений и размеры (WxH) по сплитам. */
const CompareSizeStatsSection: React.FC<Props> = ({ sizeStats, fromName, toName }) => {
  const [sizesOpen, setSizesOpen] = useState(false);
  const [selectedSplit, setSelectedSplit] = useState<string | null>(null);
  const [showAllSizes, setShowAllSizes] = useState(false);

  const formatRows = useMemo(
    () =>
      Object.entries(sizeStats.image_format_stats)
        .map(([format, delta]) => ({ label: format, delta }))
        .sort((a, b) => Math.abs(b.delta.delta) - Math.abs(a.delta.delta)),
    [sizeStats.image_format_stats],
  );

  const sizeSplitKeys = useMemo(
    () => orderSplitKeys(Object.keys(sizeStats.size_counts_per_split)),
    [sizeStats.size_counts_per_split],
  );

  const activeSplit = selectedSplit && sizeSplitKeys.includes(selectedSplit)
    ? selectedSplit
    : sizeSplitKeys[0] ?? null;

  const sizeRows = useMemo(() => {
    if (!activeSplit) return [];
    return Object.entries(sizeStats.size_counts_per_split[activeSplit] ?? {})
      .map(([size, delta]) => ({ label: size, delta }))
      .sort((a, b) => Math.abs(b.delta.delta) - Math.abs(a.delta.delta));
  }, [sizeStats.size_counts_per_split, activeSplit]);

  const visibleSizeRows = showAllSizes ? sizeRows : sizeRows.slice(0, TOP_SIZES_LIMIT);

  return (
    <section className="detail-section vcmp-section">
      <h3 className="detail-section-title">
        <i className={`fas ${ICONS.imageSize}`}></i> Форматы и размеры изображений
        <Tooltip content="Сколько изображений каждого формата файла и каждого размера в пикселях (ширина × высота) в обеих версиях.">
          <i className={`fas ${ICONS.info} vcmp-title-info`}></i>
        </Tooltip>
      </h3>

      {formatRows.length === 0 ? (
        <p className="vstats-empty">Нет данных о форматах.</p>
      ) : (
        <DeltaBarList rows={formatRows} fromLabel={fromName} toLabel={toName} headLabel="Формат" />
      )}

      {sizeSplitKeys.length > 0 && activeSplit && (
        <>
          <h5
            className="vstats-subtitle vcmp-disclosure"
            {...getDisclosureProps(sizesOpen, () => setSizesOpen(o => !o))}
          >
            <CollapseChevron open={sizesOpen} />
            <i className={`fas ${ICONS.imageSize}`}></i> Размеры (W×H) по сплитам
          </h5>
          {sizesOpen && (
            <>
              {sizeSplitKeys.length > 1 && (
                <div className="vstats-tabs">
                  {sizeSplitKeys.map(key => (
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
              {sizeRows.length === 0 ? (
                <p className="vstats-empty">Нет данных о размерах.</p>
              ) : (
                <>
                  <DeltaBarList
                    rows={visibleSizeRows}
                    fromLabel={fromName}
                    toLabel={toName}
                    headLabel="Размер"
                  />
                  {sizeRows.length > TOP_SIZES_LIMIT && (
                    <button
                      className="button secondary small vcmp-show-more"
                      onClick={() => setShowAllSizes(v => !v)}
                    >
                      {showAllSizes
                        ? 'Свернуть'
                        : `Показать все (${sizeRows.length.toLocaleString()})`}
                    </button>
                  )}
                </>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
};

export default CompareSizeStatsSection;
