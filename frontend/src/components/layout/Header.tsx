import React, { useEffect, useRef, useState } from 'react';
import { NavLink } from 'react-router-dom';
import ThemeSwitcher from './ThemeSwitcher';
import { ICONS } from '../../constants/icons';
import './Header.css';

// Пункты навигации описаны данными, чтобы не дублировать разметку
// между десктопным рядом ссылок и мобильным выпадающим меню.
const NAV_ITEMS = [
  { to: '/', label: 'Главная', icon: ICONS.home },
  { to: '/datasets', label: 'Датасеты', icon: ICONS.dataset },
  { to: '/agents', label: 'Агенты', icon: ICONS.agent },
  { to: '/models', label: 'Модели', icon: ICONS.model },
] as const;

const Header: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const navRef = useRef<HTMLDivElement>(null);

  // Тень/уплотнение шапки появляются только после прокрутки страницы.
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Закрываем меню по клику вне него и по Escape (паттерн как в ThemeSwitcher).
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (navRef.current && !navRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <header className={`header ${scrolled ? 'scrolled' : ''}`}>
      <a className="skip-link" href="#main-content">Перейти к содержимому</a>

      <div className="logo">
        <NavLink to="/" onClick={() => setOpen(false)}>KiSinWi</NavLink>
      </div>

      <div className="header-right" ref={navRef}>
        <button
          className="nav-toggle"
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? 'Закрыть меню' : 'Открыть меню'}
          aria-haspopup="true"
          aria-expanded={open}
          aria-controls="primary-nav"
        >
          <i className={`fas ${open ? ICONS.menuClose : ICONS.menu}`}></i>
        </button>

        <nav
          id="primary-nav"
          className={`nav ${open ? 'open' : ''}`}
          aria-label="Основная навигация"
        >
          {NAV_ITEMS.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => (isActive ? 'active' : '')}
              onClick={() => setOpen(false)}
            >
              <i className={`fas ${icon}`}></i>
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <ThemeSwitcher />
      </div>
    </header>
  );
};

export default Header;
