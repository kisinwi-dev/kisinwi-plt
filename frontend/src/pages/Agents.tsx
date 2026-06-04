import React, { useState } from 'react';
import { DiscussionHistory } from '../components/agents';
import './Agents.css';

type AgentsTab = 'run' | 'history';

const Agents: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AgentsTab>('history');

  return (
    <div className="agents-page">
      <div className="agents-header">
        <h1>Агенты</h1>
        <p className="agents-description">
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
          <div className="agents-run-placeholder">
            <i className="fas fa-flask"></i>
            <h3>Запуск пайплайнов</h3>
            <p>Здесь появится возможность запускать пайплайны обучения с агентами.</p>
          </div>
        ) : (
          <DiscussionHistory />
        )}
      </div>
    </div>
  );
};

export default Agents;
