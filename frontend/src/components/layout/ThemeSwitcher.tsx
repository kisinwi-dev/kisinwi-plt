import React, { useEffect, useRef, useState } from 'react';
import { THEMES, applyTheme, getStoredTheme, type ThemeId } from '../../theme/themes';
import './ThemeSwitcher.css';

const ThemeSwitcher: React.FC = () => {
  const [theme, setTheme] = useState<ThemeId>(getStoredTheme);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Закрываем меню по клику вне него.
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  const select = (id: ThemeId) => {
    applyTheme(id);
    setTheme(id);
    setOpen(false);
  };

  const current = THEMES.find((t) => t.id === theme) ?? THEMES[0];

  return (
    <div className="theme-switcher" ref={ref}>
      <button
        className="theme-switcher-toggle"
        onClick={() => setOpen((v) => !v)}
        title="Сменить тему"
        aria-haspopup="true"
        aria-expanded={open}
      >
        <i className={`fas ${current.icon}`}></i>
      </button>

      {open && (
        <ul className="theme-switcher-menu" role="menu">
          {THEMES.map((t) => (
            <li key={t.id}>
              <button
                className={`theme-switcher-item ${t.id === theme ? 'active' : ''}`}
                onClick={() => select(t.id)}
                role="menuitem"
              >
                <i className={`fas ${t.icon}`}></i>
                <span>{t.label}</span>
                {t.id === theme && <i className="fas fa-check theme-switcher-check"></i>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default ThemeSwitcher;
