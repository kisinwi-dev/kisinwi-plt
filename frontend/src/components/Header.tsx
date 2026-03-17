import React from 'react';
import { NavLink } from 'react-router-dom';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="logo">
        <NavLink to="/">KiSinWi</NavLink>
      </div>
      <nav className="nav">
        <NavLink to="/" className={({ isActive }) => isActive ? 'active' : ''}>Главная</NavLink>
        <NavLink to="/datasets" className={({ isActive }) => isActive ? 'active' : ''}>Датасеты</NavLink>
        <NavLink to="/models" className={({ isActive }) => isActive ? 'active' : ''}>Модели</NavLink>
        <NavLink to="/agents" className={({ isActive }) => isActive ? 'active' : ''}>Агенты</NavLink>
      </nav>
    </header>
  );
};

export default Header;