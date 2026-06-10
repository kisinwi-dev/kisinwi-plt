import React from 'react';
import type { DiscussionMeta } from '../../types/agentHistory';
import { DISCUSSION_STATUS_LABELS, getDiscussionTitle, getDiscussionAgents } from '../../types/agentHistory';
import { formatDateTime } from '../../utils/format';
import { useCopyToClipboard } from '../../hooks';
import { ICONS } from '../../constants/icons';
import { Tooltip } from '../common/Tooltip';

interface Props {
  discussion: DiscussionMeta;
  onSelect: (discussionId: string) => void;
  onDelete: (discussionId: string) => void;
}

const DiscussionCard: React.FC<Props> = ({ discussion, onSelect, onDelete }) => {
  const title = getDiscussionTitle(discussion);
  const copyToClipboard = useCopyToClipboard();
  const open = () => onSelect(discussion.discussion_id);
  const agentsCount = getDiscussionAgents(discussion).length;

  const handleCopyId = (e: React.MouseEvent) => {
    e.stopPropagation();
    copyToClipboard(discussion.discussion_id);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete(discussion.discussion_id);
  };

  return (
    <div
      className="card card--hoverable discussion-card"
      onClick={open}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          open();
        }
      }}
    >
      <div className="discussion-card-header">
        <div className="discussion-title-group">
          {discussion.pipeline && (
            <span className="discussion-badge">
              <i className={`fas ${ICONS.pipeline}`}></i> {discussion.pipeline}
            </span>
          )}
          <h2>{title}</h2>
          <Tooltip content="Нажмите, чтобы скопировать ID">
            <span
              className="discussion-id"
              onClick={handleCopyId}
            >
              <i className={`fas ${ICONS.id}`}></i>{discussion.discussion_id}
              <i className={`fas ${ICONS.copy} discussion-id-copy-icon`}></i>
            </span>
          </Tooltip>
        </div>
        <div className="discussion-card-actions">
          <span className={`status-badge status-${discussion.status}`}>
            {DISCUSSION_STATUS_LABELS[discussion.status]}
          </span>
          <Tooltip content="Удалить диалог">
            <button
              className="icon-button icon-button--danger"
              aria-label="Удалить диалог"
              onClick={handleDelete}
            >
              <i className={`fas ${ICONS.delete}`}></i>
            </button>
          </Tooltip>
        </div>
      </div>

      <div className="discussion-meta">
        {agentsCount > 0 && (
          <span><i className={`fas ${ICONS.agentsGroup}`}></i> Агентов: {agentsCount}</span>
        )}
        {discussion.responses_count != null && (
          <span><i className={`fas ${ICONS.discussion}`}></i> Ответов: {discussion.responses_count}</span>
        )}
        {discussion.tool_calls_count != null && discussion.tool_calls_count > 0 && (
          <span><i className={`fas ${ICONS.tools}`}></i> Вызовов: {discussion.tool_calls_count}</span>
        )}
      </div>

      <div className="discussion-meta discussion-meta--dates">
        <span><i className={`fas ${ICONS.dateCreated}`}></i> Время старта: {formatDateTime(discussion.created_at)}</span>
        {discussion.finished_at && (
          <span><i className={`fas ${ICONS.dateFinished}`}></i> Время завершения: {formatDateTime(discussion.finished_at)}</span>
        )}
      </div>

      {discussion.tags.length > 0 && (
        <div className="discussion-tags">
          <span className="discussion-tags-label">Теги:</span>
          {discussion.tags.map(tag => (
            <span key={tag} className="tag">{tag}</span>
          ))}
        </div>
      )}
    </div>
  );
};

export default DiscussionCard;
