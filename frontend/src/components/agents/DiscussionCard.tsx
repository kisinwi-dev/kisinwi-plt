import React, { useState } from 'react';
import type { DiscussionMeta, DiscussionStatus } from '../../types/agentHistory';
import { formatDateTime } from '../../utils/format';

interface Props {
  discussion: DiscussionMeta;
  onSelect: (discussionId: string) => void;
  onDelete: (discussionId: string) => void;
}

// Человекочитаемые подписи статусов дискуссии.
const STATUS_LABELS: Record<DiscussionStatus, string> = {
  active: 'Активна',
  completed: 'Завершена',
  failed: 'Ошибка',
};

// Сколько агентов показывать до сворачивания.
const AGENTS_LIMIT = 3;

const DiscussionCard: React.FC<Props> = ({ discussion, onSelect, onDelete }) => {
  const title = discussion.title ?? discussion.pipeline ?? 'Без названия';
  const [agentsExpanded, setAgentsExpanded] = useState(false);

  // Список агентов берём из агрегата; если его нет — из заявленных ролей meta.
  const agents = discussion.agents
    ?? discussion.agent_roles.map(role => ({ role, models: [] }));
  const visibleAgents = agentsExpanded ? agents : agents.slice(0, AGENTS_LIMIT);
  const hiddenCount = agents.length - visibleAgents.length;

  return (
    <div
      className="card card--hoverable discussion-card"
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
        <div className="discussion-title-group">
          <h2>{title}</h2>
          <span className="discussion-id" title={discussion.discussion_id}>
            <i className="fas fa-hashtag"></i>{discussion.discussion_id}
          </span>
        </div>
        <div className="discussion-card-actions">
          <span className={`status-badge status-${discussion.status}`}>
            {STATUS_LABELS[discussion.status]}
          </span>
          <button
            className="icon-button"
            title="Удалить диалог"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(discussion.discussion_id);
            }}
          >
            <i className="fas fa-trash"></i>
          </button>
        </div>
      </div>

      <div className="discussion-meta">
        {discussion.pipeline && (
          <span><i className="fas fa-diagram-project"></i> {discussion.pipeline}</span>
        )}
        <span><i className="fas fa-calendar-alt"></i> {formatDateTime(discussion.created_at)}</span>
        {discussion.finished_at && (
          <span><i className="fas fa-flag-checkered"></i> {formatDateTime(discussion.finished_at)}</span>
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
            {visibleAgents.map(agent => (
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
            {hiddenCount > 0 && (
              <button
                className="agents-more"
                onClick={(e) => { e.stopPropagation(); setAgentsExpanded(true); }}
              >
                +{hiddenCount} ещё
              </button>
            )}
            {agentsExpanded && agents.length > AGENTS_LIMIT && (
              <button
                className="agents-more"
                onClick={(e) => { e.stopPropagation(); setAgentsExpanded(false); }}
              >
                свернуть
              </button>
            )}
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
    </div>
  );
};

export default DiscussionCard;
