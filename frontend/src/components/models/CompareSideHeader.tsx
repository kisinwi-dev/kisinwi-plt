import React from 'react';
import { Tooltip } from '../common/Tooltip';
import type { CompareSide } from './modelCompare';

/**
 * Шапка колонки модели в таблицах сравнения: цветная полоса-акцент сверху
 * (цвет модели — тот же, что в сводке и легендах графиков), имя и версия
 * на отдельных строках, у базовой — бейдж «базовая»; ниже — опциональные
 * подстроки (children, например эпоха весов).
 */
const CompareSideHeader: React.FC<{
  side: CompareSide;
  /** Базовая модель: помечается бейджем, от неё считаются дельты. */
  isBase?: boolean;
  children?: React.ReactNode;
}> = ({ side, isBase, children }) => (
  <th className="mcmp-th-model" style={{ boxShadow: `inset 0 3px 0 0 ${side.color}` }}>
    <span className="mcmp-th-name">
      <span className="mcmp-side-dot" style={{ background: side.color }}></span>
      {side.name}
      {isBase && (
        <Tooltip content="Базовая модель: дельты у остальных моделей считаются от её значений.">
          <span className="mcmp-th-role">базовая</span>
        </Tooltip>
      )}
    </span>
    <span className="mcmp-th-version">{side.version}</span>
    {children}
  </th>
);

export default CompareSideHeader;
