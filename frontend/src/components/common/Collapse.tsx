import React from 'react';

interface CollapseChevronProps {
  open: boolean;
  className?: string;
}

/**
 * Единый индикатор сворачивания для всех вкладок.
 * Стрелка указывает вправо в свёрнутом состоянии и поворачивается вниз при раскрытии.
 * Размещается слева от заголовка (лучшая доступность при слабом зрении).
 */
export const CollapseChevron: React.FC<CollapseChevronProps> = ({ open, className }) => (
  <i
    className={`fas fa-chevron-right collapse-chevron${open ? ' open' : ''}${className ? ` ${className}` : ''}`}
    aria-hidden="true"
  />
);

/**
 * Пропсы для кликабельного заголовка сворачиваемого блока:
 * клик, клавиатура (Enter/Space), доступность (role, aria-expanded).
 * Применяется через spread, не навязывая структуру заголовка.
 */
// eslint-disable-next-line react-refresh/only-export-components
export const getDisclosureProps = (open: boolean, onToggle: () => void) => ({
  role: 'button' as const,
  tabIndex: 0,
  'aria-expanded': open,
  onClick: onToggle,
  onKeyDown: (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onToggle();
    }
  },
});
