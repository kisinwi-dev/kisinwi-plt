import React from 'react';
import type { DiscussionMeta, DiscussionStatus } from '../../types/agentHistory';
import { formatDateTime } from '../../utils/format';

interface Props {
  discussion: DiscussionMeta | null;
  discussionId: string;
}

// Человекочитаемые подписи статусов дискуссии.
const STATUS_LABELS: Record<DiscussionStatus, string> = {
  active: 'Активна',
  completed: 'Завершена',
  failed: 'Ошибка',
};

const DiscussionInfo: React.FC<Props> = ({ discussion, discussionId }) => {
  const title = discussion?.title ?? discussion?.pipeline ?? 'Без названия';
  const agents = discussion?.agents
    ?? discussion?.agent_roles.map(role => ({ role, models: [] }))
    ?? [];

  return (
    <div className="discussion-info">
      <div className="discussion-info-header">
        <div className="discussion-title-group">
          <h2 className="discussion-detail-title">{title}</h2>
          <span className="discussion-id" title={discussionId}>
            <i className="fas fa-hashtag"></i>{discussionId}
          </span>
        </div>
        {discussion && (
          <span className={`status-badge status-${discussion.status}`}>
            {STATUS_LABELS[discussion.status]}
          </span>
        )}
      </div>

      {discussion && (
        <>
          <div className="discussion-info-meta">
            {discussion.pipeline && (
              <span><i className="fas fa-diagram-project"></i> {discussion.pipeline}</span>
            )}
            <span><i className="fas fa-calendar-alt"></i> Создано: {formatDateTime(discussion.created_at)}</span>
            {discussion.finished_at && (
              <span><i className="fas fa-flag-checkered"></i> Завершено: {formatDateTime(discussion.finished_at)}</span>
            )}
            {discussion.responses_count != null && (
              <span><i className="fas fa-comments"></i> {discussion.responses_count} ответов</span>
            )}
            {discussion.tool_calls_count != null && discussion.tool_calls_count > 0 && (
              <span><i className="fas fa-wrench"></i> {discussion.tool_calls_count} вызовов</span>
            )}
          </div>

          {agents.length > 0 && (
            <div className="discussion-agents-section">
              <h4 className="discussion-section-title">
                <i className="fas fa-users"></i> Агенты ({agents.length})
              </h4>
              <div className="discussion-agents">
                {agents.map(agent => (
                  <span key={agent.role} className="agent-chip" title={agent.role}>
                    <span className="agent-chip-role">
                      <i className="fas fa-robot"></i>
                      <span className="agent-role-text">{agent.role}</span>
                    </span>
                    {agent.models.length > 0 && (
                      <span className="agent-chip-model">
                        <i className="fas fa-brain"></i> {agent.models.join(', ')}
                      </span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          )}

          {discussion.tags.length > 0 && (
            <div className="discussion-tags">
              {discussion.tags.map(tag => (
                <span key={tag} className="tag">{tag}</span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default DiscussionInfo;
