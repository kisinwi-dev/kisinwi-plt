import React from 'react';
import type { DiscussionMeta, DiscussionStatus } from '../../types/agentHistory';

interface Props {
  discussion: DiscussionMeta;
  onSelect: (discussionId: string) => void;
}

// Человекочитаемые подписи статусов дискуссии.
const STATUS_LABELS: Record<DiscussionStatus, string> = {
  active: 'Активна',
  completed: 'Завершена',
  failed: 'Ошибка',
};

const DiscussionCard: React.FC<Props> = ({ discussion, onSelect }) => {
  const title = discussion.title ?? discussion.discussion_id;

  return (
    <div
      className="discussion-card"
      onClick={() => onSelect(discussion.discussion_id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onSelect(discussion.discussion_id);
        }
      }}
    >
      <div className="discussion-card-header">
        <h2>{title}</h2>
        <span className={`status-badge status-${discussion.status}`}>
          {STATUS_LABELS[discussion.status]}
        </span>
      </div>

      <div className="discussion-meta">
        {discussion.pipeline && (
          <span><i className="fas fa-diagram-project"></i> {discussion.pipeline}</span>
        )}
        <span><i className="fas fa-calendar-alt"></i> {new Date(discussion.created_at).toLocaleString()}</span>
        {discussion.finished_at && (
          <span><i className="fas fa-flag-checkered"></i> {new Date(discussion.finished_at).toLocaleString()}</span>
        )}
      </div>

      {discussion.agent_roles.length > 0 && (
        <div className="discussion-roles">
          {discussion.agent_roles.map(role => (
            <span key={role} className="role-tag"><i className="fas fa-robot"></i> {role}</span>
          ))}
        </div>
      )}

      {discussion.tags.length > 0 && (
        <div className="discussion-tags">
          {discussion.tags.map(tag => (
            <span key={tag} className="class-tag">{tag}</span>
          ))}
        </div>
      )}
    </div>
  );
};

export default DiscussionCard;
