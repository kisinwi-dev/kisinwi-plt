import React from 'react';

interface Props {
  /** Размер стороны квадрата в px (по умолчанию 64). */
  size?: number;
  className?: string;
}

/**
 * Знак KiSinWi — глянцевый бейдж в стиле app-icon: залитый градиентный squircle,
 * белая монограмма «K», стеклянный блик и кромка. Заливка берётся из CSS-переменных темы
 * (--color-accent / --color-accent-light), поэтому бейдж адаптируется к dark / light / midnight.
 * Декоративный — aria-hidden.
 */
export const LogoMark: React.FC<Props> = ({ size = 64, className }) => {
  // Уникальные id, чтобы не конфликтовать при нескольких экземплярах.
  const uid = React.useId();
  const bg = `bg-${uid}`;
  const sheen = `sheen-${uid}`;
  const depth = `depth-${uid}`;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      className={className}
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <linearGradient id={bg} x1="6" y1="2" x2="58" y2="62" gradientUnits="userSpaceOnUse">
          <stop offset="0" style={{ stopColor: 'var(--color-accent)' }} />
          <stop offset="1" style={{ stopColor: 'var(--color-accent-light)' }} />
        </linearGradient>
        <radialGradient id={sheen} cx="20" cy="12" r="46" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#ffffff" stopOpacity="0.40" />
          <stop offset="0.55" stopColor="#ffffff" stopOpacity="0.05" />
          <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={depth} x1="0" y1="34" x2="0" y2="62" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#000000" stopOpacity="0" />
          <stop offset="1" stopColor="#000000" stopOpacity="0.20" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="17" fill={`url(#${bg})`} />
      <rect x="2" y="2" width="60" height="60" rx="17" fill={`url(#${depth})`} />
      <rect x="2" y="2" width="60" height="60" rx="17" fill={`url(#${sheen})`} />
      <rect
        x="2.6"
        y="2.6"
        width="58.8"
        height="58.8"
        rx="15.6"
        fill="none"
        stroke="#ffffff"
        strokeOpacity="0.28"
        strokeWidth="1.1"
      />
      <g stroke="#ffffff" strokeWidth="6.4" strokeLinecap="round" strokeLinejoin="round" fill="none">
        <line x1="24" y1="16" x2="24" y2="48" />
        <line x1="24" y1="33" x2="43" y2="16" />
        <line x1="24" y1="33" x2="44" y2="48" />
      </g>
    </svg>
  );
};

export default LogoMark;
