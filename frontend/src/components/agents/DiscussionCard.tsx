import React, { useState } from 'react';
import type { DiscussionMeta } from '../../types/agentHistory';
import { DISCUSSION_STATUS_LABELS, getDiscussionTitle, getDiscussionAgents } from '../../types/agentHistory';
import { formatDateTime } from '../../utils/format';
import { useCopyToClipboard } from '../../hooks';
import { CollapseChevron } from '../common/Collapse';
import { DISCUSSION_AGENTS_LIMIT } from '../../constants';

interface Props {
  discussion: DiscussionMeta;
  onSelect: (discussionId: string) => void;
  onDelete: (discussionId: string) => void;
}

const DiscussionCard: React.FC<Props> = ({ discussion, onSelect, onDelete }) => {
  const title = getDiscussionTitle(discussion);
  const [agentsExpanded, setAgentsExpanded] = useState(false);
  const [collapsed, setCollapsed] = useState(true);
  const copyToClipboard = useCopyToClipboard();

  const handleCopyId = (e: React.MouseEvent) => {
    e.stopPropagation();
    copyToClipboard(discussion.discussion_id);
  };

  // Список агентов берём из агрегата; если его нет — из заявленных ролей meta.
  const agents = getDiscussionAgents(discussion);
  const visibleAgents = agentsExpanded ? agents : agents.slice(0, DISCUSSION_AGENTS_LIMIT);
  const hiddenCount = agents.length - visibleAgents.length;

  return (
    <div className={`card discussion-card${collapsed ? ' discussion-card--collapsed' : ''}`}>
      <div className="discussion-card-header" onClick={() => setCollapsed(c => !c)} aria-expanded={!collapsed}>
        <div className="discussion-title-group">
          <CollapseChevron open={!collapsed} />
          <div className="discussion-title-inner">
            <h2
              className="discussion-card-title"
              onClick={(e) => { e.stopPropagation(); onSelect(discussion.discussion_id); }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(discussion.discussion_id);
                }
              }}
            >{title}</h2>
            <span
              className="discussion-id discussion-id--copyable"
              title="Нажмите, чтобы скопировать ID"
              onClick={handleCopyId}
            >
              <i className="fas fa-hashtag"></i>{discussion.discussion_id}
              <i className="fas fa-copy discussion-id-copy-icon"></i>
            </span>
            {collapsed && (
              <div className="discussion-card-summary">
                {discussion.pipeline && (
                  <span><i className="fas fa-diagram-project"></i> {discussion.pipeline}</span>
                )}
                <span><i className="fas fa-calendar-alt"></i> {formatDateTime(discussion.created_at)}</span>
                {discussion.finished_at && (
                  <span><i className="fas fa-flag-checkered"></i> {formatDateTime(discussion.finished_at)}</span>
                )}
              </div>
            )}
          </div>
        </div>
        <div className="discussion-card-actions">
          <span className={`status-badge status-${discussion.status}`}>
            {DISCUSSION_STATUS_LABELS[discussion.status]}
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

      {!collapsed && (
        <>
          <div className="discussion-meta">
            {discussion.pipeline && (
              <span><i className="fas fa-diagram-project"></i> {discussion.pipeline}</span>
            )}
            <span><i className="fas fa-calendar-alt"></i> {formatDateTime(discussion.created_at)}</span>
            {discussion.finished_at && (
              <span><i className="fas fa-flag-checkered"></i> {formatDateTime(discussion.finished_at)}</span>
            )}
          </div>
          {(discussion.responses_count != null || (discussion.tool_calls_count != null && discussion.tool_calls_count > 0)) && (
            <div className="discussion-meta">
              {discussion.responses_count != null && (
                <span><i className="fas fa-comments"></i> {discussion.responses_count} ответов</span>
              )}
              {discussion.tool_calls_count != null && discussion.tool_calls_count > 0 && (
                <span><i className="fas fa-wrench"></i> {discussion.tool_calls_count} вызовов</span>
              )}
            </div>
          )}

          {agents.length > 0 && (
            <div className="discussion-agents-section">
              <h4 className="discussion-section-title">
                <i className="fas fa-users"></i> Агенты ({agents.length})
              </h4>
              <div className="discussion-agents">
                {visibleAgents.map(agent => (
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
                {hiddenCount > 0 && (
                  <button
                    className="agents-more"
                    onClick={(e) => { e.stopPropagation(); setAgentsExpanded(true); }}
                  >
                    +{hiddenCount} ещё
                  </button>
                )}
                {agentsExpanded && agents.length > DISCUSSION_AGENTS_LIMIT && (
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
        </>
      )}
    </div>
  );
};

export default DiscussionCard;
