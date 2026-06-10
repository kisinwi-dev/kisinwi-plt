import React, { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
// Единый тултип приложения: hover + клавиатурный focus, aria-describedby,
// Escape для скрытия. Пузырёк рендерится порталом в body (position: fixed),
// поэтому не обрезается overflow-контейнерами модалок и таблиц.
import { createPortal } from 'react-dom';
import './Tooltip.css';

const SHOW_DELAY_MS = 250;
const EDGE_GAP = 8; // минимальный отступ пузырька от краёв вьюпорта и триггера

type Placement = 'top' | 'bottom';

interface TooltipProps {
  /** Текст подсказки. Пустой/undefined — children рендерятся без обвязки. */
  content: React.ReactNode;
  /** Базовое размещение; авто-flip, если у края вьюпорта не помещается. */
  placement?: Placement;
  /** Доп. классы обёртки-триггера (для переноса ellipsis-классов и т.п.). */
  className?: string;
  children: React.ReactNode;
}

interface BubblePos {
  top: number;
  left: number;
  placement: Placement;
  arrowLeft: number;
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  placement = 'top',
  className,
  children,
}) => {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<BubblePos | null>(null);
  const triggerRef = useRef<HTMLSpanElement>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);
  const showTimer = useRef<number | undefined>(undefined);

  const show = (immediate: boolean) => {
    window.clearTimeout(showTimer.current);
    if (immediate) {
      setOpen(true);
    } else {
      showTimer.current = window.setTimeout(() => setOpen(true), SHOW_DELAY_MS);
    }
  };

  const hide = () => {
    window.clearTimeout(showTimer.current);
    setOpen(false);
    setPos(null);
  };

  // Пузырёк измеряется уже в DOM (до этого скрыт visibility: hidden),
  // затем центрируется над триггером с клампом по краям вьюпорта.
  useLayoutEffect(() => {
    if (!open) return;
    const trigger = triggerRef.current;
    const bubble = bubbleRef.current;
    if (!trigger || !bubble) return;
    const rect = trigger.getBoundingClientRect();
    const w = bubble.offsetWidth;
    const h = bubble.offsetHeight;
    const left = Math.min(
      Math.max(rect.left + rect.width / 2 - w / 2, EDGE_GAP),
      Math.max(window.innerWidth - w - EDGE_GAP, EDGE_GAP),
    );
    let actual: Placement = placement;
    if (placement === 'top' && rect.top - h - EDGE_GAP < EDGE_GAP) actual = 'bottom';
    if (placement === 'bottom' && rect.bottom + h + EDGE_GAP > window.innerHeight) actual = 'top';
    const top = actual === 'top' ? rect.top - h - EDGE_GAP : rect.bottom + EDGE_GAP;
    const arrowLeft = Math.min(
      Math.max(rect.left + rect.width / 2 - left, 12),
      w - 12,
    );
    setPos({ top, left, placement: actual, arrowLeft });
  }, [open, placement]);

  // Пока открыт: Escape скрывает; скролл (в т.ч. внутри модалок — capture)
  // и resize тоже скрывают, чтобы пузырёк не «отвисал» от триггера.
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') hide();
    };
    const onScrollOrResize = () => hide();
    document.addEventListener('keydown', onKeyDown);
    window.addEventListener('scroll', onScrollOrResize, true);
    window.addEventListener('resize', onScrollOrResize);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('scroll', onScrollOrResize, true);
      window.removeEventListener('resize', onScrollOrResize);
    };
  }, [open]);

  useEffect(() => () => window.clearTimeout(showTimer.current), []);

  if (content === undefined || content === null || content === '') {
    return <>{children}</>;
  }

  // aria-describedby вешается на сам интерактивный элемент (кнопку/ссылку),
  // если child — единственный элемент; для текста и смешанного содержимого —
  // на обёртку-триггер.
  const childArray = React.Children.toArray(children);
  const single = childArray.length === 1 ? childArray[0] : null;
  const describedBy = open ? id : undefined;
  const childWithAria =
    single && React.isValidElement<{ 'aria-describedby'?: string }>(single)
      ? React.cloneElement(single, { 'aria-describedby': describedBy })
      : null;

  return (
    <span
      ref={triggerRef}
      className={`tooltip-trigger${className ? ` ${className}` : ''}`}
      aria-describedby={childWithAria ? undefined : describedBy}
      onMouseEnter={() => show(false)}
      onMouseLeave={hide}
      onFocus={() => show(true)}
      onBlur={hide}
    >
      {childWithAria ?? children}
      {open &&
        createPortal(
          <div
            ref={bubbleRef}
            id={id}
            role="tooltip"
            className="tooltip-bubble"
            data-placement={pos?.placement ?? placement}
            style={
              pos
                ? { top: pos.top, left: pos.left }
                : { top: 0, left: 0, visibility: 'hidden' }
            }
          >
            {content}
            <span
              className="tooltip-arrow"
              style={pos ? { left: pos.arrowLeft } : undefined}
            />
          </div>,
          document.body,
        )}
    </span>
  );
};
