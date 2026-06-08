import React, { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { DiscussionHistory, RunPipelineForm } from '../components/agents';
import './Agents.css';

type AgentsTab = 'run' | 'history';

const Agents: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AgentsTab>('history');
  const [, setSearchParams] = useSearchParams();

  // После старта пайплайна — переключаемся на историю и открываем дискуссию.
  const handleStarted = (discussionId: string) => {
    setActiveTab('history');
    setSearchParams({ discussion: discussionId });
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
      <div className="agents-tabs">
        <button
          className={`agents-tab ${activeTab === 'run' ? 'active' : ''}`}
          onClick={() => setActiveTab('run')}
        >
          <i className="fas fa-play"></i> Запуск
        </button>
        <button
          className={`agents-tab ${activeTab === 'history' ? 'active' : ''}`}
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
