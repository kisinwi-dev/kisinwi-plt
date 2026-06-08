import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/theme.css';
import './styles/components.css';
import App from './App.tsx'
import { applyTheme, getStoredTheme } from './theme/themes';

// Применяем сохранённую тему до рендера, чтобы не было мигания.
applyTheme(getStoredTheme());

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);