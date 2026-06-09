import React from 'react';
import type { DiscussionMeta } from '../../types/agentHistory';
import { DISCUSSION_STATUS_LABELS, getDiscussionTitle, getDiscussionAgents } from '../../types/agentHistory';
import { formatDateTime } from '../../utils/format';
import { ICONS } from '../../constants/icons';

interface Props {
  discussion: DiscussionMeta | null;
  discussionId: string;
}

const DiscussionInfo: React.FC<Props> = ({ discussion, discussionId }) => {
  const title = discussion ? getDiscussionTitle(discussion) : 'Без названия';
  const agents = discussion ? getDiscussionAgents(discussion) : [];

  return (
    <div className="discussion-info">
      <div className="discussion-info-header">
        <div className="discussion-title-group">
          <div className="discussion-title-inner">
            <h2 className="discussion-detail-title">{title}</h2>
            <span className="discussion-id" title={discussionId}>
              <i className={`fas ${ICONS.id}`}></i>{discussionId}
            </span>
          </div>
        </div>
        {discussion && (
          <span className={`status-badge status-${discussion.status}`}>
            <span className={`status-dot ${discussion.status === 'active' ? 'status-dot--pulse' : ''}`} aria-hidden="true" />
            {DISCUSSION_STATUS_LABELS[discussion.status]}
          </span>
        )}
      </div>

      {discussion && (
        <>
          {(discussion.responses_count != null ||
            (discussion.tool_calls_count != null && discussion.tool_calls_count > 0)) && (
            <div className="discussion-stats">
              {discussion.responses_count != null && (
                <div className="stat-chip">
                  <i className={`fas ${ICONS.discussion}`}></i>
                  <span className="stat-chip-value">{discussion.responses_count}</span>
                  <span className="stat-chip-label">ответов</span>
                </div>
              )}
              {discussion.tool_calls_count != null && discussion.tool_calls_count > 0 && (
                <div className="stat-chip">
                  <i className={`fas ${ICONS.tools}`}></i>
                  <span className="stat-chip-value">{discussion.tool_calls_count}</span>
                  <span className="stat-chip-label">вызовов</span>
                </div>
              )}
            </div>
          )}

          <div className="discussion-info-meta">
            {discussion.pipeline && (
              <span><i className={`fas ${ICONS.pipeline}`}></i> {discussion.pipeline}</span>
            )}
            <span><i className={`fas ${ICONS.dateCreated}`}></i> Создано: {formatDateTime(discussion.created_at)}</span>
            {discussion.finished_at && (
              <span><i className={`fas ${ICONS.dateFinished}`}></i> Завершено: {formatDateTime(discussion.finished_at)}</span>
            )}
          </div>

          {agents.length > 0 && (
            <div className="discussion-agents-section">
              <h4 className="discussion-section-title">
                <i className={`fas ${ICONS.agentsGroup}`}></i> Агенты ({agents.length})
              </h4>
              <div className="discussion-agents">
                {agents.map(agent => (
                  <div key={agent.role} className="agent-card">
                    <span className="agent-card-avatar">
                      <i className={`fas ${ICONS.agent}`}></i>
                    </span>
                    <div className="agent-card-body">
                      <span className="agent-card-role" title={agent.role}>{agent.role}</span>
                      {agent.models.length > 0 && (
                        <div className="agent-card-models">
                          {agent.models.map(model => (
                            <span key={model} className="agent-model-badge" title={model}>
                              <i className={`fas ${ICONS.agentModel}`}></i>
                              <span className="agent-model-text">{model}</span>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
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
