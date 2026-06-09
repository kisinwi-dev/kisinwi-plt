import { BrowserRouter, Routes, Route } from 'react-router-dom';
// BrowserRouter — обёртка для приложения, которая обеспечивает работу маршрутизации
// Routes и Route — компоненты для декларативного описания маршрутов.

import { NotificationProvider } from './contexts/NotificationContext';
// Провайдер контекста уведомлений. Он хранит глобальное состояние списка уведомлений
// и предоставляет функции для показа/скрытия.

import NotificationToast from './components/notification/NotificationToast';
// Компонент, который отображает всплывающие уведомления (тосты) в правом верхнем углу.
// Он подписывается на контекст уведомлений и рисует все активные уведомления.

import Header from './components/layout/Header';
import Footer from './components/layout/Footer';

import Home from './pages/Home';
import Datasets from './pages/Datasets';
import DatasetDetail from './pages/DatasetDetail';
import Models from './pages/Models';
import ModelDetail from './pages/ModelDetail';
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
              <Route path="/datasets/:id" element={<DatasetDetail />} />
              <Route path="/models" element={<Models />} />
              <Route path="/models/:id" element={<ModelDetail />} />
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