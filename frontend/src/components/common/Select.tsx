import React, { useEffect, useRef, useState } from 'react';

export interface SelectOption {
  value: string;
  label: string;
}

interface Props {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  /** Иконка-префикс (класс Font Awesome), показывается слева от значения. */
  icon?: string;
  /** Подпись для пустого значения (первый пункт списка). */
  placeholder?: string;
  /** ARIA-метка для кнопки. */
  ariaLabel?: string;
}

/**
 * Кастомный выпадающий список в стилистике фронтенда.
 * Заменяет нативный <select>, чьё всплывающее окно рисуется ОС и не вписывается в тему.
 * Поддержка клавиатуры: Enter/Space/↓ — открыть, ↑/↓ — навигация, Enter — выбор, Esc — закрыть.
 */
const Select: React.FC<Props> = ({ value, options, onChange, icon, placeholder, ariaLabel }) => {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const rootRef = useRef<HTMLDivElement>(null);

  const allOptions: SelectOption[] = placeholder
    ? [{ value: '', label: placeholder }, ...options]
    : options;
  const selected = allOptions.find((o) => o.value === value) ?? allOptions[0];

  // Закрытие по клику вне компонента.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, [open]);

  const openMenu = () => {
    setActiveIndex(Math.max(0, allOptions.findIndex((o) => o.value === value)));
    setOpen(true);
  };

  const choose = (v: string) => {
    onChange(v);
    setOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
        e.preventDefault();
        openMenu();
      }
      return;
    }
    switch (e.key) {
      case 'Escape':
        e.preventDefault();
        setOpen(false);
        break;
      case 'ArrowDown':
        e.preventDefault();
        setActiveIndex((i) => Math.min(allOptions.length - 1, i + 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setActiveIndex((i) => Math.max(0, i - 1));
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        if (allOptions[activeIndex]) choose(allOptions[activeIndex].value);
        break;
    }
  };

  return (
    <div className={`select${open ? ' select--open' : ''}`} ref={rootRef}>
      <button
        type="button"
        className="select-trigger"
        onClick={() => (open ? setOpen(false) : openMenu())}
        onKeyDown={handleKeyDown}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
      >
        {icon && <i className={icon}></i>}
        <span className="select-value">{selected?.label}</span>
        <i className="fas fa-chevron-down select-caret"></i>
      </button>

      {open && (
        <ul className="select-menu" role="listbox">
          {allOptions.map((o, i) => (
            <li
              key={o.value || '__empty'}
              role="option"
              aria-selected={o.value === value}
              className={`select-option${o.value === value ? ' select-option--selected' : ''}${i === activeIndex ? ' select-option--active' : ''}`}
              onMouseEnter={() => setActiveIndex(i)}
              onClick={() => choose(o.value)}
            >
              <span className="select-option-label">{o.label}</span>
              {o.value === value && <i className="fas fa-check select-option-check"></i>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Select;
