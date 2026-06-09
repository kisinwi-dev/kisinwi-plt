import React from 'react';
import { Navigate, useNavigate, useParams } from 'react-router-dom';
import { DiscussionDetail } from '../components/agents';
import './Agents.css';

// Отдельная страница детального просмотра дискуссии (/agents/discussion/:discussionId).
// Сама загрузка meta и ленты сообщений — внутри DiscussionDetail.
const AgentDiscussion: React.FC = () => {
  const { discussionId } = useParams<{ discussionId: string }>();
  const navigate = useNavigate();

  if (!discussionId) {
    return <Navigate to="/agents" replace />;
  }

  return (
    <div className="page">
      <DiscussionDetail discussionId={discussionId} onBack={() => navigate('/agents')} />
    </div>
  );
};

export default AgentDiscussion;
