import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DiscussionHistory, RunPipelineForm } from '../components/agents';
import './Agents.css';

type AgentsTab = 'run' | 'history';

const Agents: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AgentsTab>('history');
  const navigate = useNavigate();

  // После старта пайплайна — открываем страницу дискуссии.
  const handleStarted = (discussionId: string) => {
    navigate(`/agents/discussion/${discussionId}`);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1>Агенты</h1>
        <p className="page-description">
          Запускайте пайплайны обучения с агентами и просматривайте историю их работы.
        </p>
      </div>

      {/* Переключатель вкладок */}
      <div className="page-tabs">
        <button
          className={`page-tab ${activeTab === 'run' ? 'active' : ''}`}
          onClick={() => setActiveTab('run')}
        >
          <i className="fas fa-play"></i> Запуск
        </button>
        <button
          className={`page-tab ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <i className="fas fa-clock-rotate-left"></i> История
        </button>
      </div>

      {/* Содержимое вкладки */}
      <div className="agents-content">
        {activeTab === 'run' ? (
          <RunPipelineForm onStarted={handleStarted} />
        ) : (
          <DiscussionHistory />
        )}
      </div>
    </div>
  );
};

export default Agents;
