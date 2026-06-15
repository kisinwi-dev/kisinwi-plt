// Уважение к системной настройке «уменьшить движение».
// CSS-анимации гасятся через @media (prefers-reduced-motion: reduce);
// для JS-анимаций (плавный скролл) проверяем ту же настройку здесь.
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined'
    && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
}
