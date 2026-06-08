import React from 'react';
import { NavLink } from 'react-router-dom';
import ThemeSwitcher from './ThemeSwitcher';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="logo">
        <NavLink to="/">KiSinWi</NavLink>
      </div>
      <div className="header-right">
        <nav className="nav">
          <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>Главная</NavLink>
          <NavLink to="/datasets" className={({ isActive }) => isActive ? 'active' : ''}>Датасеты</NavLink>
          <NavLink to="/agents" className={({ isActive }) => isActive ? 'active' : ''}>Агенты</NavLink>
          <NavLink to="/models" className={({ isActive }) => isActive ? 'active' : ''}>Модели</NavLink>
        </nav>
        <ThemeSwitcher />
      </div>
    </header>
  );
};

export default Header;
