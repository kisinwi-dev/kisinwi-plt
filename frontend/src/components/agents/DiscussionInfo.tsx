import React from 'react';
import type { DiscussionMeta } from '../../types/agentHistory';
import { DISCUSSION_STATUS_LABELS, getDiscussionTitle, getDiscussionAgents } from '../../types/agentHistory';
import { formatDateTime } from '../../utils/format';

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
              <i className="fas fa-hashtag"></i>{discussionId}
            </span>
          </div>
        </div>
        {discussion && (
          <span className={`status-badge status-${discussion.status}`}>
            {DISCUSSION_STATUS_LABELS[discussion.status]}
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
                  <div key={agent.role} className="agent-card">
                    <span className="agent-card-avatar">
                      <i className="fas fa-robot"></i>
                    </span>
                    <div className="agent-card-body">
                      <span className="agent-card-role" title={agent.role}>{agent.role}</span>
                      {agent.models.length > 0 && (
                        <div className="agent-card-models">
                          {agent.models.map(model => (
                            <span key={model} className="agent-model-badge" title={model}>
                              <i className="fas fa-brain"></i>
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
