import { BrowserRouter, Routes, Route } from 'react-router-dom';
// BrowserRouter — обёртка для приложения, которая обеспечивает работу маршрутизации
// Routes и Route — компоненты для декларативного описания маршрутов.

import { NotificationProvider } from './contexts/NotificationContext';
// Провайдер контекста уведомлений. Он хранит глобальное состояние списка уведомлений
// и предоставляет функции для показа/скрытия.

import NotificationToast from './components/notification/NotificationToast';
// Компонент, который отображает всплывающие уведомления (тосты) в правом верхнем углу.
// Он подписывается на контекст уведомлений и рисует все активные уведомления.

import Header from './components/Header';
import Footer from './components/Footer';

import Home from './pages/Home';
import Datasets from './pages/Datasets';
import Models from './pages/Models';
import Agents from './pages/Agents';

import './styles/App.css';

function App() {
  return (
    <NotificationProvider>
      <BrowserRouter>
        <div className="app-wrapper">
          <Header />
          <main className="main-content">
            <Routes>
              {/*
                Route определяет соответствие между путём URL и компонентом,
                который должен отображаться.
              */}
              <Route path="/" element={<Home />} />
              <Route path="/datasets" element={<Datasets />} />
              <Route path="/models" element={<Models />} />
              <Route path="/agents" element={<Agents />} />
            </Routes>
          </main>
          <Footer />
          <NotificationToast />
        </div>
      </BrowserRouter>
    </NotificationProvider>
  );
}

export default App;