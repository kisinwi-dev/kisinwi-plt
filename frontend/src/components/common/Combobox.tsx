import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ICONS } from '../../constants/icons';
import './Combobox.css';

interface Props {
  value: string;
  onChange: (value: string) => void;
  /** Подсказки для выпадающего списка. Ввод при этом остаётся свободным. */
  options: string[];
  placeholder?: string;
  /** Иконка-префикс (класс Font Awesome). */
  icon?: string;
  disabled?: boolean;
  id?: string;
}

/**
 * Поле с автодополнением: можно выбрать из списка (выпадашка в стиле фронтенда,
 * как кастомный Select) либо ввести значение вручную. По мере ввода список
 * фильтруется по подстроке.
 */
const Combobox: React.FC<Props> = ({ value, onChange, options, placeholder, icon, disabled, id }) => {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const rootRef = useRef<HTMLDivElement>(null);

  // Фильтрация по подстроке; если значение точно совпадает с пунктом — показываем весь список.
  const filtered = useMemo(() => {
    const q = value.trim().toLowerCase();
    if (!q || options.some((o) => o.toLowerCase() === q)) return options;
    return options.filter((o) => o.toLowerCase().includes(q));
  }, [value, options]);

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

  const choose = (v: string) => {
    onChange(v);
    setOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!open) { setOpen(true); setActiveIndex(0); return; }
      setActiveIndex((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((i) => Math.max(0, i - 1));
    } else if (e.key === 'Enter') {
      if (open && filtered[activeIndex]) {
        e.preventDefault();
        choose(filtered[activeIndex]);
      }
    } else if (e.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <div className={`combobox${open ? ' combobox--open' : ''}`} ref={rootRef}>
      <div className="combobox-control">
        {icon && <i className={`${icon} combobox-icon`}></i>}
        <input
          id={id}
          type="text"
          className="combobox-input"
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          autoComplete="off"
          onChange={(e) => { onChange(e.target.value); setOpen(true); setActiveIndex(0); }}
          onFocus={() => setOpen(true)}
          onKeyDown={handleKeyDown}
        />
        {options.length > 0 && (
          <button
            type="button"
            className="combobox-caret"
            tabIndex={-1}
            disabled={disabled}
            aria-label="Показать варианты"
            onClick={() => setOpen((o) => !o)}
          >
            <i className={`fas ${ICONS.expand}`}></i>
          </button>
        )}
      </div>

      {open && filtered.length > 0 && (
        <ul className="combobox-menu" role="listbox">
          {filtered.map((o, i) => (
            <li
              key={o}
              role="option"
              aria-selected={o === value}
              className={`combobox-option${o === value ? ' combobox-option--selected' : ''}${i === activeIndex ? ' combobox-option--active' : ''}`}
              onMouseEnter={() => setActiveIndex(i)}
              onMouseDown={(e) => { e.preventDefault(); choose(o); }}
            >
              <span className="combobox-option-label">{o}</span>
              {o === value && <i className={`fas ${ICONS.selected} combobox-option-check`}></i>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default Combobox;
