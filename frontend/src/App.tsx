import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { NotificationProvider } from './contexts/NotificationContext';
import NotificationToast from './components/notification/NotificationToast';
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
import ErrorBoundary from './components/common/ErrorBoundary';

// Страницы грузятся лениво: каждая — отдельный чанк, тяжёлые либы (antd, recharts,
// react-markdown) не попадают в начальный бандл, а подтягиваются при переходе на роут.
const Home = lazy(() => import('./pages/Home'));
const Datasets = lazy(() => import('./pages/Datasets'));
const DatasetDetail = lazy(() => import('./pages/DatasetDetail'));
const DatasetCompare = lazy(() => import('./pages/DatasetCompare'));
const Models = lazy(() => import('./pages/Models'));
const ModelDetail = lazy(() => import('./pages/ModelDetail'));
const ModelCompare = lazy(() => import('./pages/ModelCompare'));
const Agents = lazy(() => import('./pages/Agents'));
const AgentDiscussion = lazy(() => import('./pages/AgentDiscussion'));

import './styles/App.css';

// Содержимое внутри роутера: отдельный компонент, чтобы вызвать useLocation.
// key={pathname} перемонтирует ErrorBoundary при смене роута — иначе после краша
// рендера hasError остаётся true навсегда и навигация по ссылкам не спасает.
function Layout() {
  const { pathname } = useLocation();
  return (
    <div className="app-wrapper">
      <Header />
      <main id="main-content" className="main-content">
        <ErrorBoundary key={pathname}>
          <Suspense fallback={<div className="route-loading">Загрузка…</div>}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/datasets" element={<Datasets />} />
              <Route path="/datasets/:id" element={<DatasetDetail />} />
              <Route path="/datasets/:id/compare" element={<DatasetCompare />} />
              <Route path="/models" element={<Models />} />
              <Route path="/models/compare" element={<ModelCompare />} />
              <Route path="/models/:id" element={<ModelDetail />} />
              <Route path="/agents" element={<Agents />} />
              <Route path="/agents/discussion/:discussionId" element={<AgentDiscussion />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </main>
      <Footer />
      <NotificationToast />
    </div>
  );
}

function App() {
  return (
    <NotificationProvider>
      <BrowserRouter>
        <Layout />
      </BrowserRouter>
    </NotificationProvider>
  );
}

export default App;